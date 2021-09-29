from django.shortcuts import render
from .forms import InputForm
from django_ffmpeg.models import ConvertingCommand
from sound_prediction.functions import *
from django.http import HttpResponse
from os import path
from pydub import AudioSegment

def homepage(request):
    return render(request, 'homepage.html', {})


def give_prediction(request):
    URL = request.POST.session.get('url')

    return render(request, 'PredictionDone.html', URL)


# Create your views here.
def graph_(file_name):
        y, sr = librosa.load(file_name, mono=True, duration=5)
        cmap = plt.get_cmap('inferno')
        # plt.figure(figsize=(8, 8))
        a = plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off')
        plt.savefig('sound_prediction/static/temp_heatmap.png')
        plt.clf()
        plt.close()

def login(request):
    URL = ""
    context={}
    if request.method == "POST":
        # Get the posted form
        MyLoginForm = InputForm(request.POST)
        print(MyLoginForm)
        if MyLoginForm.is_valid():
            URL = MyLoginForm.cleaned_data['URL']
            FEVER =  MyLoginForm.cleaned_data['FEVER']
            OC = MyLoginForm.cleaned_data['OC']
            health_status = MyLoginForm.cleaned_data['health_status']
    else:
        MyLoginForm = InputForm()
    ext = URL[-4:]
    if (ext !=".wav"):
        # files
        # AudioSegment.converter = 'Lib/site-packages/django_ffmpeg'
        # wget.download(URL, 'sound_prediction/files/temp' + ext)
        # src = 'sound_prediction/files/temp' + ext
        # dst = 'sound_prediction/files/temp' + '.wav'
        # sound = AudioSegment.from_mp3(src)
        # sound.export(dst, format="wav")
        pass
    else:
        #Save the downloaded audio
        wget.download(URL, 'sound_prediction/files/temp'+ext)

    #save the generated heatmap
    file_name = 'sound_prediction/files/temp'+'.wav'
    context['graph'] = graph_(file_name)
    data_ = [['sound_prediction/files/temp', health_status, 'sound_prediction/files/temp'+'.wav',int(FEVER),int(OC),'Coswara']]

    def feature_extractor(file_name_,row):
        name = row[0]

        audio, sr = librosa.load(file_name_, mono=True, )
        print(audio)
        print(sr)
        # For MFCCS
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        print(mfccsscaled)

        # Mel Spectogram
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr)
        s_db = librosa.power_to_db(melspec, ref=np.max)
        librosa.display.specshow(s_db)

        savepath = os.path.join('', name + '.png')
        pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
        pylab.close()

        return mfccsscaled, savepath

    features = []
    diagnoses = []
    imgpaths = []

    for row in (data_):
        mfccs,savepath  = feature_extractor(file_name,row)
        print(mfccs)
        features.append(mfccs)
        imgpaths.append(savepath)
        diagnoses.append([row[3],row[4]])

    isnone = lambda x: x is not None
    label = lambda x: 1 if x == 'positive_mild' or x == 'positive_moderate' or x == 'COVID-19' else 0
    cast_x = list(map(isnone, features))
    data_y = list(map(label, health_status))

    data_x = [features[i] for i in range(len(features)) if cast_x[i] == True]
    data_xx = [imgpaths[i] for i in range(len(imgpaths)) if cast_x[i] == True]
    data_xp = [diagnoses[i] for i in range(len(diagnoses)) if cast_x[i] == True]
    data_y = [data_y[i] for i in range(len(features)) if cast_x[i] == True]

    assert len(data_x) == len(data_xx) == len(data_xp), "Data lengths do not match"

    indices = np.arange(len(data_x))
    NUM_shuf = 5
    DATA = {i: {} for i in range(NUM_shuf)}

    for i in range(NUM_shuf):
        np.random.shuffle(indices)

        DATA[i]['MFCCS'] = np.array([data_x[i] for i in indices])
        DATA[i]['MEL'] = [data_xx[i] for i in indices]
        DATA[i]['EXTRA'] = np.array([data_xp[i] for i in indices])
        DATA[i]['LABELS'] = np.array([data_y[i] for i in indices])

    class CustomDataset(tf.keras.utils.Sequence):
        def __init__(self, imgfiles, labels, batch_size, target_size=(64, 64), shuffle=False, scale=255, n_classes=1,
                     n_channels=3):
            self.batch_size = batch_size
            self.dim = target_size
            self.labels = labels
            self.imgfiles = imgfiles
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.n_channels = n_channels
            self.scale = scale

            self.c = 0
            self.on_epoch_end()

        def __len__(self):
            # returns the number of batches
            return int(np.floor(len(self.imgfiles) / self.batch_size))

        def __getitem__(self, index):
            # returns one batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Generate data
            X, y = self.__data_generation(indexes)
            return X, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.imgfiles))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                img = cv2.imread(self.imgfiles[ID])
                img = cv2.resize(img, self.dim, interpolation=cv2.INTER_CUBIC)
                X[i,] = img / self.scale

                # Store class
                y[i] = self.labels[ID]

                self.c += 1
            return X, y  # keras.utils.to_categorical(y, num_classes=self.n_classes)

    class CustomPipeline(tf.keras.utils.Sequence):
        def __init__(self, data_x, data_y, batch_size=1, shuffle=False, n_classes=1):
            self.features = data_x
            self.labels = data_y
            self.batch_size = 1
            self.shuffle = shuffle
            self.n_features = self.features.shape[1]
            self.n_classes = 1
            self.on_epoch_end()

        def __len__(self):
            return int(np.floor(len(self.features) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            X, y = self.__data_generation(indexes)
            return X, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.features))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            X = np.empty((self.batch_size, self.n_features))
            y = np.empty((self.batch_size), dtype=int)

            for i, ID in enumerate(indexes):
                X[i,] = self.features[ID]
                y[i,] = self.labels[ID]
            return X, y

    class MultipleInputGenerator(tf.keras.utils.Sequence):
        """Wrapper of two generatos for the combined input model"""

        def __init__(self, X1, X2, Y, batch_size, target_size=(64, 64)):
            self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size, shuffle=False)
            self.genX2 = CustomDataset(X2, Y, batch_size=batch_size, shuffle=False, target_size=target_size)

        def __len__(self):
            return self.genX1.__len__()

        def __getitem__(self, index):
            X1_batch, Y_batch = self.genX1.__getitem__(index)
            X2_batch, Y_batch = self.genX2.__getitem__(index)
            X_batch = [X1_batch, X2_batch]
            return X_batch, Y_batch

    class TripleInputGenerator(tf.keras.utils.Sequence):
        """Wrapper of two generatos for the combined input model"""

        def __init__(self, X1, X2, X3, Y, batch_size, target_size=(64, 64)):
            self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size, shuffle=False)
            self.genX2 = CustomDataset(X2, Y, batch_size=batch_size, shuffle=False, target_size=target_size)
            self.genX3 = CustomPipeline(X3, Y, batch_size=batch_size, shuffle=False)

        def __len__(self):
            return self.genX1.__len__()

        def __getitem__(self, index):
            X1_batch, Y_batch = self.genX1.__getitem__(index)
            X2_batch, Y_batch = self.genX2.__getitem__(index)
            X3_batch, Y_batch = self.genX3.__getitem__(index)

            X_batch = [X1_batch, X2_batch, X3_batch]
            return X_batch, Y_batch

    iii = 0
    test_features = DATA[iii]['MFCCS']
    test_extra = DATA[iii]['EXTRA']
    test_imgs = DATA[iii]['MEL']
    test_labels = DATA[iii]['LABELS']
    TEST = TripleInputGenerator(test_features, test_imgs, test_extra, test_labels, batch_size=1, target_size=(64, 64))

    model = keras.models.load_model('sound_prediction/model/020--0.569--0.411.hdf5')
    y_score = model.predict(TEST)
    y_score = str(y_score[0])
    print('y_score:',y_score)

    try:
        path = 'sound_prediction/files'
        os.remove(path)
        print("% s removed successfully" % path)
    except OSError as error:
        print(error)
        print("File path can not be removed")

    return render(request, 'PredictionDone.html', {"URL": URL,'y_score': y_score })
