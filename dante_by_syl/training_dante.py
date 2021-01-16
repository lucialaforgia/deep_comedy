import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use('seaborn-whitegrid')

def draw_graphs(history, logs_dir):

    epochs = len(history['loss'])
    x = np.arange(0, epochs) + 1

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title("Training results: epoch vs accuracy")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.plot(x, history['accuracy'])
    ax1.plot(x, history['val_accuracy'])
    ax1.legend(['train_accuracy', 'val_accuracy'], loc='center right')
#        ax1.set_ylim(0.75,1)
    ax1.tick_params(axis='y' )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    color = 'tab:red'
    ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, history['lr'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig(os.path.join(logs_dir, "accuracy.png"))

    plt.clf()


    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title("Training results: epoch vs loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(x, history['loss'])
    ax1.plot(x, history['val_loss'])
    ax1.legend(['train_loss', 'val_loss'], loc='center right')
#        ax1.set_ylim(0.75,1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    color = 'tab:red'
    ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, history['lr'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig(os.path.join(logs_dir, "loss.png"))

    plt.clf()

def train_model(working_dir, model, model_filename, dataset_train, dataset_val, epochs=50):


    # Path where the final model will be saved
    models_dir = os.path.join(working_dir, 'models')
    os.makedirs(models_dir, exist_ok = True) 
    model_file = os.path.join(models_dir, model_filename+".h5")

    # Path where the logs will be saved
    logs_dir = os.path.join(working_dir, 'logs', model_filename)
    shutil.rmtree(logs_dir, ignore_errors=True)
    os.makedirs(logs_dir, exist_ok = True) 

    log_file = os.path.join(logs_dir, "training_logs.csv")

    # Path where the tensorboard logs will be saved
    tb_logs_dir = os.path.join(logs_dir, 'tensorboard')
    os.makedirs(tb_logs_dir, exist_ok = True) 

    # Plot the model structure
    model_image_file = os.path.join(logs_dir, model_filename+'.png')
    tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True, show_layer_names=True)

    model_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss', 
        mode='auto', 
        verbose=1, 
        )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        restore_best_weights=True, 
        patience=5,
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
        patience=2, 
        min_lr=0.0005, 
        verbose=1
        )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logs_dir,
        histogram_freq=1
        )

    print("TRAINING MODEL")
    history = model.fit(dataset_train,
        epochs=epochs, 
        validation_data=dataset_val,
        callbacks=[
            model_callback,
            csv_logger_callback, 
            reduce_lr_callback,
            early_stopping_callback,
            tensorboard_callback
            ]
        )

    print("TRAINING COMPLETE!")
    
    draw_graphs(history.history, logs_dir)

    return history
