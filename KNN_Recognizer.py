import threading
from queue import Queue, Full
from array import array
import pyaudio, wave,os
from threading import Thread
import time
import random
from KNN import knn
from speaker_rec import who_speaks
            
dataDire = os.getcwd()+'/trainPickles/'

os.chdir(os.getcwd()+'/Biip')

audio = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 3072

MIN_VOLUME = 1000

MIN_SECONDS = 0.5

MAX_SECONDS = 3

MAX_SILENCE = 0.3



chunks_max = int(MAX_SECONDS*RATE/CHUNK_SIZE)+1
chunks_silence_max = int(MAX_SILENCE*RATE/CHUNK_SIZE)+1


q_ecouter_enregistrer = Queue(maxsize=-1)
q_enregistrer_knner = Queue(maxsize=-1)


def index_microphone():
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range (0,numdevices):
        if audio.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0 and 'ebcam' in audio.get_device_info_by_host_api_device_index(0,i).get('name'):
            return(i)

   
def SaveToWav(data, name):
    waveFile = wave.open(name, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(data))
    waveFile.close()


def listen(stopped, q):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
 	input_device_index=index_microphone(),
        channels=CHANNELS,
        rate=RATE,
        input=True,
	output=False,
        frames_per_buffer=CHUNK_SIZE,
    )

    while True:
        if stopped.wait(timeout=0):
            break
        try:
            q.put(array('h', stream.read(CHUNK_SIZE)))
        except Full:
            pass  # discard


def record(stopped, q):
    ind=1
    data=[]
    silence_list_test=[]
    origin=time.time()
    start=time.time()
    print('Now you can talk ...')
    while True:
        if stopped.wait(timeout=0):
            break
        chunk = q.get()
        vol = max(chunk)
        if vol >= MIN_VOLUME:
            data.append(chunk)
            duration_file=int(len(data)*CHUNK_SIZE/RATE*100)/100
            if duration_file>=MAX_SECONDS:
                file_name=str(int((start-origin)*100)/100) +"-"+ str(int((start-origin+MAX_SECONDS)*100)/100)+".wav"
                print(str(ind)+'  '+file_name)
                ind+=1
                print("ignoring noise")
                start=start+MAX_SECONDS
                SaveToWav(data, file_name)
                q_enregistrer_knner.put(file_name)
                data=[]
            for j in range(chunks_silence_max):
                silence_list_test.append(q.get())
            if all(max(x)<MIN_VOLUME for x in silence_list_test):
                for k in range(2*chunks_silence_max//3):
                    data.append(silence_list_test[k])
                duration_file=int(len(data)*CHUNK_SIZE/RATE*100)/100
                if duration_file>=MIN_SECONDS:
                    file_name=str(int((start-origin)*100)/100) +"-"+ str(int((start-origin+duration_file)*100)/100)+".wav"
                    print(str(ind)+'  '+file_name)
                    ind+=1
                    print()
                    SaveToWav(data, file_name)
                    q_enregistrer_knner.put(file_name)
                data=[]
            for j in range(chunks_silence_max):
                q.put(silence_list_test[j])
            if not all(max(x)<MIN_VOLUME for x in silence_list_test) and max(silence_list_test[0])<MIN_VOLUME:
                x=silence_list_test[0]
                i=0
                while max(x)<MIN_VOLUME:
                    data.append(x)
                    i+=1
                    x=silence_list_test[i]
                for j in range(i):
                    surplus=q.get()
            silence_list_test=[]
        else:
            start=time.time()
            data=[]


def main():

    stopped = threading.Event()

    listen_t = threading.Thread(target=listen, args=(stopped, q_ecouter_enregistrer))
    record_t = threading.Thread(target=record, args=(stopped, q_ecouter_enregistrer))
    listen_t.start()
    record_t.start()

    try:
        while True:
            listen_t.join(0.1)
            record_t.join(0.1)
           
    except KeyboardInterrupt:
        stopped.set()

    listen_t.join()
    record_t.join()



class ConsumerThread(Thread):
    def run(self):
        while True:
            aud = q_enregistrer_knner.get()
            print(knn(aud,dataDire, 30, 8)[2])
            #who_speaks(aud)
            os.remove(aud)
            q_enregistrer_knner.task_done()


if __name__ == '__main__':
    try:
        ConsumerThread().start()
        main()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        sys.exit()
