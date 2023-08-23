import numpy as np
import pandas as pd

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def get_data(validate,train_df,PARAMS):
  """ pandas Dataframe 형식 데이터를 Temporal Fusion Transformer Dataset 형식 맞게 변환

    Args:
        validate (_type_): _description_
        train_df (_type_): _description_
        PARAMS (_type_): _description_

    Returns:
        _type_: train_dataset, validation data_set
  """
  train_df = train_df.astype(np.float64)
  
  train_df['order'] = train_df['order'].astype(int)
  
  PARAMS['KNOW_IN_FUTURE'] = [x for x in train_df.columns if x!='target']
  print("know in future variables : ", PARAMS['KNOW_IN_FUTURE'])
  
  tr_ds = TimeSeriesDataSet(
          # with validate=False use all data
          train_df[:-128] if validate else train_df,
          time_idx="order",
          target='target',
          group_ids=['build_idx'],
          weight=None,
          min_encoder_length=64,
          max_encoder_length=PARAMS['MAX_ENCODER_LENGTH'],
          min_prediction_length=64,
          max_prediction_length=128, #predict 500m

          time_varying_known_reals=PARAMS['KNOW_IN_FUTURE'], #시간 지나도 아는 실수 변수들

          static_reals=[], #고정된 실수들

          time_varying_unknown_categoricals=[], #시간 지나면 모르는 string들 (우린 없음)
          time_varying_unknown_reals=['target'], #시간 지나면 모르는 정수들

          add_relative_time_idx=True,  # add as feature
          add_target_scales=True,  # add as feature
          add_encoder_length=True,  # add as feature
      )

  va_ds = None
  if validate:
      # validation dataset not used for submission
      va_ds = TimeSeriesDataSet.from_dataset(
          tr_ds, train_df, predict=True, stop_randomization=True
      )
  return tr_ds, va_ds



# training
def fit(seed, tr_ds, va_loader=None, cur_loss=MAPE() ,PARAMS=None):
    
    # create dataloaders for model
    tr_loader = tr_ds.to_dataloader(
        train=True, batch_size=PARAMS['BATCH_SIZE'], num_workers=2
    )


    if va_loader is not None:
        # stop training, when loss metric does not improve on validation set
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=20,
            verbose=True,
            mode="min"
        )
        lr_logger = LearningRateMonitor(logging_interval="epoch")  # log the learning rate
        callbacks = [lr_logger, early_stopping_callback]
    else:
        # gather 10 checkpoints with best traing loss
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=PARAMS['CKPTROOT'],
            filename=f'seed={seed}'+'-{epoch:03d}-{train_loss:.2f}',
            save_top_k=3 #10으로 하면 앙상블
        )
        callbacks = [checkpoint_callback]


    # create trainer
    trainer = pl.Trainer(
        max_epochs=PARAMS['epoch'], #40
        gradient_clip_val=PARAMS['gradient_clip_val'],
        limit_train_batches=60,
        callbacks=callbacks,
        logger=TensorBoardLogger(PARAMS['LOGDIR'])
    )

    # use pre-deterined leraning rate schedule for final submission
    learning_rate = PARAMS['learning_rate']

    # initialise model with pre-determined hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        tr_ds,
        learning_rate=learning_rate,
        hidden_size=PARAMS['hidden_size'],
        attention_head_size=PARAMS['attention_head_size'],
        dropout=PARAMS['dropout'],
        hidden_continuous_size=PARAMS['hidden_continuous_size'],
        output_size=1,
        loss=cur_loss,
        log_interval=10,  # log example every 10 batches
        logging_metrics=[MAPE()], #,myMAPE().to('cuda')],
        reduce_on_plateau_patience=4,  # reduce learning automatically
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    kwargs = {'train_dataloaders': tr_loader}
    if va_loader:
        kwargs['val_dataloaders'] = va_loader

    # fit network
    trainer.fit(
        tft,
        **kwargs
    )

    return


def forecast(ckpt, train_df, test_df):
    """ckpt 주소에 해당하는 모델로 결과 예측 후 return

    Args:
        ckpt (_type_): _description_
        train_df (_type_): _description_
        test_df (_type_): _description_

    Returns:
        _type_: tensor
    """
    # load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    max_encoder_length = best_tft.dataset_parameters['max_encoder_length']
    max_prediction_length = best_tft.dataset_parameters['max_prediction_length']

    # use 5 weeks of training data at the end
    encoder_data = train_df[-max_encoder_length :]

    # get last entry from training data
    last_data = train_df.iloc[[-1]]

    # fill NA target value in test data with last values from the train dataset
    target_cols = [c for c in test_df.columns if 'target' in c]
    for c in target_cols:
        test_df.loc[:, c] = last_data[c].item()

    decoder_data = test_df

    # combine encoder and decoder data. decoder data is to be predicted
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)


    new_prediction_data = new_prediction_data.astype(np.float64)

    pred = best_tft.predict(new_prediction_data , mode="raw", return_x=True)

    return pred.output.prediction



def validation(model_path, train_df, test_df, PARAMS):
        """
        validation score 확인을 위한 예측 수행, pd.dataframe형식 return

        Args:
            model_path (_type_): _description_
            train_df (_type_): _description_
            test_df (_type_): _description_
            PARAMS (_type_): _description_

        Returns:
            _type_: _description_
        """

        PARAMS['KNOW_IN_FUTURE'] = [x for x in train_df.columns if x!='target']

        tr_ds, va_ds = get_data( validate=True , train_df = train_df, PARAMS=PARAMS)
        va_loader = va_ds.to_dataloader(
                train=False, batch_size=50, num_workers=2
            )
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        predictions = model.predict(va_loader)

        temp = predictions.view(100,-1)
        
        return temp