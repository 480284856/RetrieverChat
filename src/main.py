import os
import time
import json
import random
import string
import logging
import asyncio
import datetime
import threading
import speech_recognition as sr

from typing import List
from pygame import mixer
from asyncio import Task
from zijie_tts import tts
from langchain_chroma import Chroma
from speech_recognition import AudioData
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.text import TextLoader
from ali_stt_voice_awake import Recognition,lingji_stt_gradio_va
from ali_stt_voice_awake_async import lingji_stt_gradio_va_async,transform_res,recognition

# 语音唤醒-激活到休眠使用的全局变量
global to_work,flag_time_to_sleep,dida,task_main_workflow,stopper_trans_res_monitor
flag_time_to_sleep=False

task_main_workflow:Task    
stopper_trans_res_monitor = threading.Event()

# 语音模块
async def lingji_stt_gradio_va2(inputs):
    transcript = await lingji_stt_gradio_va_async()
    return transcript

# 检索小模型
def audio_produce(documents:List):
    '''把document的page_content转换成语音，然后再把路径存储在metadata中。'''
    for doc in documents:
        audio_path = tts(doc.page_content)
        doc.metadata['audio_path'] = audio_path

# 语音播报
def get_audio_path(documents:List):
    return documents[0].metadata['audio_path']

def play_audio(audio_path):
    mixer.init()
    mixer.music.load(audio_path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.001)
    
    mixer.music.unload()
    mixer.quit()  

def get_chain(embedding_model_url, name_embedding_model):
    # text
    text_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'responses.txt')
    text_loader = TextLoader(text_path)
    data = text_loader.load()

    # document
    text_splitter = CharacterTextSplitter(separator='\n\n', chunk_size=0, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    
    # add audio
    audio_produce(all_splits)

    # embedding model loading
    local_embeddings = OllamaEmbeddings(base_url=embedding_model_url, model=name_embedding_model)

    # vector store
    vector_store = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    # retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})

    # chain
    chain = lingji_stt_gradio_va2 | retriever
    chain = chain | get_audio_path | play_audio 

    return chain

def get_logger():
    # 日志收集器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 设置控制台处理器，当logger被调用时，控制台处理器额外输出被调用的位置。
    # 创建一个控制台处理器并设置级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 创建一个格式化器，并设置格式包括文件名和行号
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(ch)

    return logger

class VoiceAwake():
    def __init__(self) -> None:
        self.logger = get_logger()
        self.recognition = Recognition(model='paraformer-realtime-v2',
                          format='wav',
                          sample_rate=16000,
                          callback=None)
        
        self.welcome_audio_path = None

    def save_audio_file(self, audio:AudioData, sample_rate=16000):
            file_name = self.get_random_file_name()
            with open(file_name, "wb") as f:
                f.write(audio.get_wav_data(convert_rate=sample_rate))
            return file_name
    
    def recognize_speech(self, audio_path, recognizer:Recognition) ->str:
            recognition_result = recognizer.call(audio_path).get_sentence()
            if recognition_result:
                return recognition_result[0]['text']
            else:
                return ''
            
    def get_random_file_name(self, length=20, extension='.wav'):
        '''create a random file name with current time'''
        current_time = datetime.datetime.now().strftime("%Y-%m-%d%H-%M-%S")
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        return f"{current_time}_{random_string}{extension}"
    
    def play_welcome_audio(self):
        if self.welcome_audio_path is None:
            self.welcome_audio_path = tts("诶！")
        play_audio(self.welcome_audio_path)

    def is_awaked(self,):
        recognizer = sr.Recognizer()
        
        # 麦克风准备
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, 1)            # 调整背景噪音
            
            while True:
                try:
                    '''一直监听语音'''
                    self.logger.info("Listening for wake word 'hei siri'...")
                    audio = recognizer.listen(source, timeout=3)                 # 监听麦克风
                    self.logger.info("Recognizing done.")
                    audio_path = self.save_audio_file(audio)
                    result = self.recognize_speech(audio_path, recognizer=self.recognition)
                    os.remove(audio_path)
                    self.logger.info(f"Recognized: {result}")
                    
                    '''当用户说出特定唤醒词时'''
                    if "你好" in result:
                        self.logger.info("Wake word detected!")

                        '''TODO: 给出固定的欢迎回复'''
                        self.play_welcome_audio()
                        return True
                except Exception as e:
                    self.logger.error(f"Error occurred: {e}")
    
    def is_awaked_v2(self,):
        """使用 lingji_stt_gradio_va2 来判断是否应该被唤醒"""

        """
        背景(2024-8-27)：speech_recognition 库的监听API太过灵敏，很容易把杂音认为是人在说话，效果不好，而灵积API则不会。

        使用灵积API，一直监听环境中的语音，如果监听到唤醒词，就播放欢迎音，并返回True。
        """
        while True:
            try:
                transcript = lingji_stt_gradio_va()
                self.logger.info(f"is_awaked_v2: transcript: {transcript}")
                if "你好" in transcript:
                    self.play_welcome_audio()
                    self.logger.info("Wake word detected!")
                    return True
            except Exception as e:
                print(f"Error occurred: {e}")

def dida_timer():
    global flag_time_to_sleep

    flag_time_to_sleep=True

async def run_main_workflow(
        main_workflow
):
    try:
        await main_workflow.ainvoke(None)
    except Exception as e:
        print(f"main_workflow error: {e}")

def transform_res_monitor(logger):
    global to_work, flag_time_to_sleep,dida,task_main_workflow,stopper_trans_res_monitor,recognition
    
    while not stopper_trans_res_monitor.is_set():
        if not flag_time_to_sleep and transform_res['sentence']:
            if dida.is_alive():
                dida.cancel()   
                logger.info("transform_res_monitor: 检测到语音输入，停止计时。")
            break
        elif flag_time_to_sleep:
            to_work = False
            if not task_main_workflow.cancelled():
                task_main_workflow.cancel()
            recognition.stop()    # 异步任务强制结束，相关资源可能没有释放
            logger.info("transform_res_monitor: 因长时间没有语音输入，进入休眠状态。")
            break
        else:
            time.sleep(0.01)

async def main(
        logger,
        silence_shreshold:int=10,
):
    global to_work, dida, task_main_workflow, flag_time_to_sleep

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')) as F:
        args = json.load(F)
        embedding_model_url = args['embedding_model_url']
        name_embedding_model = args['name_embedding_model']

    main_workflow = get_chain(embedding_model_url, name_embedding_model)

    is_awaked = VoiceAwake()
    
    while True:
        logger.info("进入休眠状态。。。")
        if is_awaked.is_awaked_v2():
            to_work = True
           
            while to_work:
                try:
                    
                    task_main_workflow = asyncio.create_task(run_main_workflow(main_workflow))
                    
                    dida = threading.Timer(silence_shreshold, dida_timer)
                    thread_transform_res_monitor = threading.Thread(target=transform_res_monitor, args=(logger,))

                    dida.start()
                    thread_transform_res_monitor.start()
                    try:
                        await task_main_workflow
                    except asyncio.CancelledError:
                        logger.info("Task has been cancelled")
                    # Create an asyncio task
                    flag_time_to_sleep = False
                except Exception as e:
                    logger.info("发生了错误：{e}, 重新进行服务。。。".format(e))

                    if task_main_workflow.exception():
                        task_main_workflow.cancel()
                    if dida.is_alive():
                        dida.cancel()
                    to_work = True

if __name__ == '__main__':
    logger = get_logger()
    asyncio.run(main(logger))