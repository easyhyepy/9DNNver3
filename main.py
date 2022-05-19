import tensorflow as tf

## Mnist 데이터셋을 불러옴 ##
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## reshape and normalization
# 모델의 input으로 넣기 위해 (28 * 28 = 784) 데이터 형태로 변형, 0~1 사이의 실수 값으로 정규화
x_train = x_train.reshape((60000, 28 * 28)) / 255.0
x_test = x_test.reshape((10000, 28 * 28)) / 255.0

model = tf.keras.models.Sequential()  # Sequential model

DICT = {}

def StackingLayers(a,b,c):
    model.add(tf.keras.layers.Dense(a, activation='relu', input_shape=(28*28,)))
    model.add(tf.keras.layers.Dense(b, activation='relu'))
    model.add(tf.keras.layers.Dense(c, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

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
    model.fit(x_train, y_train, epochs=epc, verbose=1, validation_split=0.01)    #epochs변경, verbose가 0이면 깔끔하게 출력, 1이면 쪼금 자세히 1로해서내

def PrintAccuracy(opt, lr, epc):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\n테스트 정확도 : ', test_acc)
    DICT[str(opt)+'_'+str(lr)+'_'+str(epc)] = test_acc



optimizer = ['adagrad'] #xiver 있다던데 'sgd', 'adam', 'rmsprop', 'adadelta','adagrad'
loss = ['sparse_categorical_crossentropy'] #, 'mse', 'binary_crossentropy', 'categorical_crossentropy']  #'BinaryCrossentropy class', 'CategoricalCrossentropy class', 'SparseCategoricalCrossentropy class', 'Poisson class', 'binary_crossentropy function', 'categorical_crossentropy function', 'sparse_categorical_crossentropy function', 'poisson function', 'KLDivergence class', 'kl_divergence function'
lr = [0.1]
epc = [21]

layer1=[770]
layer2=[420, 300, 540, 420, 300, 540]
layer3=[70]

for k in optimizer:
    for i in lr:
      for j in epc:
          for a in layer1:
              for b in layer2:
                  for c in layer3:
                    StackingLayers(a,b,c)
                    model.summary()
                    ModelCompile(k, i)
                    ModelFit(j)
                    PrintAccuracy(k, i, j)  #원래빈칸

                    #아래 내용을 추가해주지 않으면 기존 모델 재활용되기에, 이상하게 꼬여서 초기화가 필요하다고 생각했음. 그래서 위 코드를 붙였더니 모델이 각자 다른게 생성되는 것을 확인 (이름(숫자) 다름) 일일이 돌렸을 때와 값이 거의 동일한 것을 확인함.
                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                    x_train = x_train.reshape((60000, 28 * 28)) / 255.0
                    x_test = x_test.reshape((10000, 28 * 28)) / 255.0

                    model = tf.keras.models.Sequential()  # Sequential model

print(DICT)  #잘 출력됨. {'sgd_0.01_1': 0.9067000150680542, 'adam_0.01_1': 0.9531000256538391, 'rmsprop_0.01_1': 0.8880000114440918, 'adadelta_0.01_1': 0.7645000219345093, 'adagrad_0.01_1': 0.9251999855041504}
import operator
SORT= sorted(DICT.items(), key=operator.itemgetter(1), reverse=True)
print(SORT)