import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def draw_graphs(history, logs_dir):

    epochs = len(history['loss'])
    x = np.arange(0, epochs) + 1

    fig, ax1 = plt.subplots()

    plt.title("Training results: epoch vs accuracy")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.plot(x, history['accuracy'])
    ax1.plot(x, history['val_accuracy'])
    ax1.legend(['train_accuracy', 'val_accuracy'], loc='center right')
#        ax1.set_ylim(0.75,1)
    ax1.tick_params(axis='y' )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, history['lr'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig(os.path.join(logs_dir, "accuracy.png"))

    plt.clf()


    fig, ax1 = plt.subplots()

    plt.title("Training results: epoch vs loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(x, history['loss'])
    ax1.plot(x, history['val_loss'])
    ax1.legend(['train_loss', 'val_loss'], loc='center right')
#        ax1.set_ylim(0.75,1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, history['lr'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig(os.path.join(logs_dir, "loss.png"))

    plt.clf()

def train_model(working_dir, model, dataset_train, dataset_val, epochs=100, batch_size=32):

    # Directory where the checkpoints will be saved
    #checkpoint_dir = os.path.join(working_dir, 'training_checkpoints')
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch:03d}")

    # Path where the final model will be saved
    models_dir = os.path.join(working_dir, 'models')
    shutil.rmtree(models_dir, ignore_errors=True)
    os.makedirs(models_dir, exist_ok = True) 
    model_file = os.path.join(models_dir, "dante_by_char_model_final.h5")
#    model_epoch_file = os.path.join(models_dir, "dante_by_char_model_epoch_{epoch:03d}.h5")
    best_model_file = os.path.join(models_dir, "dante_by_char_best_model.h5")

    # Path where the logs will be saved
    logs_dir = os.path.join(working_dir, 'logs')
    shutil.rmtree(logs_dir, ignore_errors=True)
    os.makedirs(logs_dir, exist_ok = True) 

    log_file = os.path.join(logs_dir, "dante_by_char.csv")

    # Path where the tensorboard logs will be saved
    tb_logs_dir = os.path.join(logs_dir, 'tensorboard')
    os.makedirs(tb_logs_dir, exist_ok = True) 

    #checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    #    filepath=checkpoint_prefix,
    #    save_weights_only=False,
    #    monitor='loss', 
    #    mode='auto', 
    #    verbose=1, 
    #    )

    model_callback=tf.keras.callbacks.ModelCheckpoint(
#        filepath=model_epoch_file,
        filepath=best_model_file,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss', 
        mode='auto', 
        verbose=1, 
        )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        restore_best_weights=True, 
        patience=15,
        monitor='val_loss', 
        mode='auto', 
        verbose=1
        )

    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        log_file, 
        separator=';', 
        append=False,
        )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        mode='auto', 
        factor=0.5, 
        patience=3, 
        min_lr=0.0001, 
        verbose=1
        )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logs_dir,
        histogram_freq=1
        )

    print("TRAINING MODEL")
#    history = model.fit(x_train, y_train,
    history = model.fit(dataset_train,
        epochs=epochs, 
#        batch_size=batch_size, 
        validation_data=dataset_val,
        callbacks=[
#            checkpoint_callback,
            model_callback,
            csv_logger_callback, 
            reduce_lr_callback,
            early_stopping_callback,
            tensorboard_callback
            ]
        )

    print("TRAINING COMPLETE!")
    
    model.save(model_file)
#    print(history.history)

    draw_graphs(history.history, logs_dir)

#    loss = history.history['loss'][-1]
#    val_loss = history.history['val_loss'][-1]
#    acc = history.history['accuracy'][-1]
#    val_acc = history.history['val_accuracy'][-1]
#    print("LOSS: {:5.2f}".format(loss) + " - ACC: {:5.2f}%".format(100 * acc) + " - VAL_LOSS: {:5.2f}".format(val_loss) + " - VAL_ACC: {:5.2f}%".format(100 * val_acc))
    return history
