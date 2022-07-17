import numpy as np
import time


def run_model(model_setting, data_group, saved_path, logger):
    model_key = model_setting.model_key
    attn_key = model_setting.attn_key
    batch_control = model_setting.batch_control
    train_x_shape = data_group.train_x_matrix.shape
    train_y_shape = data_group.train_y_matrix.shape
    in_shape = train_x_shape[1:]
    num_classes = train_y_shape[1]
    training_generator = None
    class_batch_size = 20
    if batch_control:
        logger.info("batch control confirm: " + str(batch_control))
        train_x_matrix = data_group.train_x_matrix
        train_y_vector = data_group.train_y_vector
        train_y_matrix = to_categorical(train_y_vector, len(np.unique(train_y_vector)))
        class_batch_size = ret_class_batch_size(model_setting.batch_size, data_group.train_y_vector)
        # if class_batch_size < 5:
        #     class_batch_size = 5
        logger.info("new batch size: " + str(class_batch_size))

        batch_size = class_batch_size * num_classes
        model_setting.batch_size = batch_size
        x_train_list, y_train_list = class_input_process(train_x_matrix, train_y_vector, train_y_matrix, num_classes)
        training_generator = MiBatchGenerator(x_train_list, y_train_list, class_batch_size, True)
    print("2: " + str(data_group.train_x_matrix.shape))

    # apply_se = True
    if model_key == 'tapnet':
        model = TapNetBuild(num_classes, in_shape, attn_key, model_setting.ga_sigma, class_batch_size)
    else:
        model = ModelBuild(num_classes, in_shape, model_key, attn_key, model_setting.ga_sigma, class_batch_size)
    model.build((None,) + in_shape)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logger.info(short_model_summary)
    return model_training(model, data_group, saved_path, logger, model_setting, training_generator)


def model_training(model, data_group, saved_path, logger, cnn_setting=None, generator=None):
    epochs = 50
    batch_size = 128
    learning_rate = 1e-3
    monitor = 'loss'
    optimization_mode = 'auto'
    if cnn_setting is not None:
        epochs = cnn_setting.max_iter
        batch_size = cnn_setting.batch_size
        learning_rate = cnn_setting.learning_rate
    train_x_matrix = data_group.train_x_matrix
    train_y_vector = data_group.train_y_vector
    test_x_matrix = data_group.test_x_matrix
    test_y_vector = data_group.test_y_vector

    classes = np.unique(train_y_vector)
    le = LabelEncoder()
    y_ind = le.fit_transform(train_y_vector.ravel())
    recip_freq = len(train_y_vector) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    logger.info("Class weights : " + str(class_weight))
    print("Class weights : ", class_weight)
    class_w_dict = {}
    class_index = 0
    for item in class_weight:
        class_w_dict[class_index] = item
        class_index = class_index + 1
    train_y_matrix = to_categorical(train_y_vector, len(np.unique(train_y_vector)))
    test_y_matrix = to_categorical(test_y_vector, len(np.unique(test_y_vector)))

    factor = 1. / np.cbrt(2)

    model_checkpoint = ModelCheckpoint(saved_path, verbose=1, mode=optimization_mode, monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=2000, mode=optimization_mode, factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]
    # callback_list = [model_checkpoint]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    if cnn_setting.model_key == 'tapnet' and model.attn_type=='tap_attn':
        model.add_loss(lambda: 0.01 * model.loss_term)
    if generator is None:
        start_time = time.time()
        log_history = model.fit(train_x_matrix, train_y_matrix, batch_size=batch_size, epochs=epochs, callbacks=callback_list, class_weight=class_w_dict, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        training_time = time.time() - start_time
    else:
        start_time = time.time()
        # log_history = model.fit_generator(generator=generator, epochs=epochs, callbacks=callback_list, class_weight=class_w_dict, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        log_history = model.fit_generator(generator=generator, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        training_time = time.time() - start_time
    return log_history, model, training_time


def ret_class_batch_size(batch_size, train_y_vector, version=0):
    unique, counts = np.unique(train_y_vector, return_counts=True)
    train_size = len(train_y_vector)
    num_iter = train_size/float(batch_size)
    if num_iter == int(num_iter):
        num_iter = int(num_iter)
    else:
        num_iter = int(num_iter) + 1
    max_class_len = max(counts)
    class_batch_size = float(max_class_len)/num_iter
    if class_batch_size != int(class_batch_size):
        class_batch_size = int(class_batch_size) + 1
    else:
        class_batch_size = int(class_batch_size)
    return class_batch_size

