import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from glob import glob
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from json import dump, load
import tensorflow as tf
import multiprocessing as mp
from keras_tqdm import TQDMCallback
from tqdm import tqdm



def gen_lenet(img_height, img_width, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(img_height, img_width, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def gen_vgg(img_height, img_width, num_classes):
    base_model = VGG19(weights=None, include_top=False, input_shape=(img_height, img_width, 1))

    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    x = layer_dict['block2_pool'].output
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def gen_vgg16(img_height, img_width, num_classes):
    base_model = VGG16(weights=None, include_top=False, input_shape=(img_height, img_width, 1))

    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    x = layer_dict['block2_pool'].output
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def gen_inception(img_height, img_width, num_classes):
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(img_height, img_width, 1))

    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    x = layer_dict['mixed10'].output
    # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def gen_resnet(img_height, img_width, num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 1))

    x = base_model.layers[-1].output
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


models = [gen_lenet, gen_vgg, gen_vgg16, gen_inception, gen_resnet]
models_names = ['LeNet', 'VGG19', 'VGG16', 'InceptionV3', 'ResNet50']

names_funcs = dict(zip(models_names, models))


def run_model(name, img_size, num_classes, results, curve_values, i, X_train, y_train, X_test, y_test, train_form,
              test_form):
    strategy = tf.distribute.MirroredStrategy()
    gpus = strategy.num_replicas_in_sync
    print("Starting ", name, ' k=', 1, ' devices=', gpus, sep='')
    with strategy.scope():
        model = names_funcs[name](*img_size, num_classes)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, verbose=2, batch_size=64 * gpus, validation_data=(X_test, y_test),
                        callbacks=[TQDMCallback()])
    print("Evaluating")
    predict_train = model.predict(X_train)
    np.save(train_form % (name, i), predict_train)
    predict_test = model.predict(X_test)
    np.save(test_form % (name, i), predict_test)
    _, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f'{name} fold {i}:', test_acc)
    results[name][i] = test_acc
    curve_values[name][i] = {k: np.array(v).tolist() for k, v in history.history.items()}
    with open(RESULTS_FILE, 'w') as fp:
        dump(results, fp)
    with open(CURVE_FILE, 'w') as fp:
        dump(curve_values, fp)
    with open(MODEL_FORM % (name, i), 'w') as fp:
        fp.write(model.to_json())
    model.save_weights(WEIGTHS_FORM % (name, i))


with open('config.json') as fp:
    config = load(fp)

X_files = config['X_files']
y_files = config['y_files']
WEIGTHS_FORM = config['WEIGTHS_FORM']
MODEL_FORM = config['MODEL_FORM']
RESULTS_FILE = config['RESULTS_FILE']
CURVE_FILE = config['CURVE_FILE']
RAW_PREDICT_TRAIN_FORM = config.get('RAW_PREDICT_TRAIN_FORM', 'predict_train_%s_%d.npy')
RAW_PREDICT_TEST_FORM = config.get('RAW_PREDICT_TEST_FORM', 'predict_test_%s_%d.npy')

train_files = sorted(glob(config['train_files']))
test_files = sorted(glob(config['test_files']))

try:
    with open(RESULTS_FILE) as fp:
        results = load(fp)
except:
    results = {}
    open(RESULTS_FILE, 'w').close()

try:
    with open(CURVE_FILE) as fp:
        curve_values = load(fp)
except:
    curve_values = {}
    open(CURVE_FILE, 'w').close()

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    for name, model_gen, X_file, y_file in tqdm(zip(models_names, models, X_files, y_files), desc="Geral"):
        X = np.load(X_file)
        y = np.load(y_file)
        img_size = X.shape[1], X.shape[2]
        num_classes = len(np.unique(y))
        results.setdefault(name, {})
        curve_values.setdefault(name, {})
        for i, (train_file, test_file) in tqdm(enumerate(zip(train_files, test_files), 1), desc='K-fold'):
            i = str(i)
            if i in results.get(name, {}):
                print(f'In file {name} fold {i}:', results[name][i])
                continue
            train = np.load(train_file)
            test = np.load(test_file)
            proc = ctx.Process(target=run_model, name=f"Run model {name}", args=(name, img_size, num_classes, results,
                                                                             curve_values, i, X[train], y[train],
                                                                             X[test], y[test], RAW_PREDICT_TRAIN_FORM,
                                                                             RAW_PREDICT_TEST_FORM))
            proc.start()
            proc.join()
            with open(RESULTS_FILE) as fp:
                results = load(fp)
            print("Results", name, "len", len(results[name]))
            with open(CURVE_FILE) as fp:
                curve_values = load(fp)
            print("Curve Values", name, "len", len(results[name]))
