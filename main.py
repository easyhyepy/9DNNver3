import tensorflow as tf

## Mnist 데이터셋을 불러옴 ##
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## reshape and normalization
# 모델의 input으로 넣기 위해 (28 * 28 = 784) 데이터 형태로 변형, 0~1 사이의 실수 값으로 정규화
x_train = x_train.reshape((60000, 28 * 28)) / 255.0
x_test = x_test.reshape((10000, 28 * 28)) / 255.0

model = tf.keras.models.Sequential()  # Sequential model

def StackingLayers():
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(28*28,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  #activation 변경

def ModelCompile(opt, lr):
    print(opt, lr)
    if (opt == 'sgd'):
        opt = tf.optimizers.SGD(learning_rate=lr)
    elif (opt == 'adam'):
        opt = tf.optimizers.Adam(learning_rate=lr)
    elif (opt == 'rmsprop'):
        opt = tf.optimizers.RMSprop(learning_rate=lr)
    elif (opt == 'adadelta'):
        opt = tf.optimizers.Adadelta(learning_rate=lr)
    elif (opt == 'adagrad'):
        opt = tf.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])    #optimizer 변경 (loss는 건듦x)

def ModelFit(epc):
    print("epc: ", epc)
    model.fit(x_train, y_train, epochs=epc, verbose=1, validation_split=0.2)    #epochs변경, verbose가 0이면 깔끔하게 출력, 1이면 쪼금 자세히

def PrintAccuracy():
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\n테스트 정확도 : ', test_acc)


optimizer = ['sgd', 'adam', 'rmsprop', 'adadelta','adagrad'] #xiver 있다던데
loss = ['sparse_categorical_crossentropy'] #, 'mse', 'binary_crossentropy', 'categorical_crossentropy']  #'BinaryCrossentropy class', 'CategoricalCrossentropy class', 'SparseCategoricalCrossentropy class', 'Poisson class', 'binary_crossentropy function', 'categorical_crossentropy function', 'sparse_categorical_crossentropy function', 'poisson function', 'KLDivergence class', 'kl_divergence function'
lr = [0.01]
epc = [1, 5, 10]

for k in optimizer:
    for i in lr:
      for j in epc:
        StackingLayers()
        model.summary()
        ModelCompile(k, i)
        ModelFit(j)
        PrintAccuracy()


        #아래 내용을 추가해주지 않으면 기존 모델 재활용되기에, 이상하게 꼬여서 초기화가 필요하다고 생각했음. 그래서 위 코드를 붙였더니 모델이 각자 다른게 생성되는 것을 확인 (이름(숫자) 다름)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((60000, 28 * 28)) / 255.0
        x_test = x_test.reshape((10000, 28 * 28)) / 255.0

        model = tf.keras.models.Sequential()  # Sequential model
