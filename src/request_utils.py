import requests
from requests_toolbelt import MultipartEncoder
import os
import re
import uuid
import datetime
import pytz

from .file_utils import audiobyte_to_numpy, save_audio_file_from_numpy

GET_MODEL_LIST_METHOD = "model_list"
GET_MODEL_DISPLAY_NAME_METHOD = "model_display_name"
PERFORM_TTS_METHOD = "perform_tts"
PERFORM_VC_METHOD = "perform_vc"
GET_EMBED_LIST_METHOD = "embed_list"
GET_LANGUAGE_LIST_METHOD = "language_list"
TIMEZONE = "Asia/Seoul"

# cpu-tts-models/src/text_utils/multi_lingual_g2p.py 참고
### validMultiLingualLanguages
language_display_name = {
    "ml-english" : "영어",
    "ml-korean" : "한국어",
    "ml-english-korean" : "영어-한국어",
    "ml-spanish" : "스페인어",
    "ml-japanese" : "일본어",
    "ml-chinese" : "중국어",
    "ml-vietnamese" : "베트남어",
    "ml-french" : "프랑스어",
    "ml-arabic" : "아랍어",
    "ml-javanese" : "자바어",
}
display_name_language = {display_name : language for language, display_name in language_display_name.items()}



class RequestManager():
    def __init__(self, server_path, server_port, bearer_token, tmp_save_dir="/home/usr/audio_outputs"):
        self.server_path = server_path
        self.server_port = server_port
        self.bearer_token = bearer_token # , auth=BearerAuth(self.bearer_token)
        self.tmp_save_dir = tmp_save_dir
        self.seoul_tz = pytz.timezone(TIMEZONE)

        self.model_display_name_dict = {}

    def get_server_tts_model_list(self):
        model_list_response = requests.get(
            f"{self.server_path}:{self.server_port}/{GET_MODEL_DISPLAY_NAME_METHOD}",
            headers = {"Authorization": f"Bearer {self.bearer_token}"},
        )
        model_display_name_dict = {key : value for key, value in eval(model_list_response.content).items() if value.startswith("vits")}
        self.model_display_name_dict.update(model_display_name_dict)
        return list(model_display_name_dict.keys())

    def get_server_vc_model_list(self):
        model_list_response = requests.get(
            f"{self.server_path}:{self.server_port}/{GET_MODEL_DISPLAY_NAME_METHOD}",
            headers = {"Authorization": f"Bearer {self.bearer_token}"},
        )
        model_display_name_dict = {key : value for key, value in eval(model_list_response.content).items() if value.startswith("vc")}
        self.model_display_name_dict.update(model_display_name_dict)
        return list(model_display_name_dict.keys())
    
    def get_server_embed_list(self, model_name):
        embed_list_response = requests.get(
            f"{self.server_path}:{self.server_port}/{GET_EMBED_LIST_METHOD}",
            json={"model_name" : self.model_display_name_dict[model_name]},
            headers = {"Authorization": f"Bearer {self.bearer_token}"},
        )
        embed_list = eval(embed_list_response.content)
        return embed_list

    def get_server_language_list(self, model_name):
        language_list_response = requests.get(
            f"{self.server_path}:{self.server_port}/{GET_LANGUAGE_LIST_METHOD}",
            json={"model_name" : self.model_display_name_dict[model_name]},
            headers = {"Authorization": f"Bearer {self.bearer_token}"},
        )
        language_list = eval(language_list_response.content)
        return [language_display_name[ll] for ll in language_list]

    def get_filepath(self, model_name, text):
        idx = 1
        filepath = f"{self.tmp_save_dir}/gradio-{model_name}_{text[:10]}_{datetime.datetime.now(self.seoul_tz).strftime('%Y%m%d_%H%M%S')}.wav"
        while True:
            filepath = re.sub(r'\s+', ' ', filepath)
            if os.path.isfile(filepath):
                filepath = f"{self.tmp_save_dir}/gradio-{model_name}_{text[:10]}_{datetime.datetime.now(self.seoul_tz).strftime('%Y%m%d_%H%M%S')} ({idx}).wav"
                idx += 1
            else:
                break
        return filepath

    def save_audio_file(self, model_name, text, audio_component):
        tmp_filepath = self.get_filepath(model_name, text)
        sample_rate, audio_numpy = audio_component
        save_audio_file_from_numpy(tmp_filepath, audio_numpy, sample_rate)
        return tmp_filepath

    def perform_vc(
        self,
        audio_file,
        model_name,
        embed,
        pitch,
        noise_scale,
        sampling_rate,
        audio_clearance,
    ):
        try:
            m = MultipartEncoder(
                fields={
                    "audio_file" : (os.path.basename(audio_file), open(audio_file, 'rb'), 'text/plain'), 
                    "model_name" : self.model_display_name_dict[model_name], 
                    "embed" : embed, 
                    "pitch": str(pitch), 
                    "noise_scale": str(noise_scale), 
                    "sampling_rate":str(sampling_rate), 
                    "audio_clearance":str(audio_clearance),
                }
            )
            audiobyte_response = requests.get(
                f"{self.server_path}:{self.server_port}/{PERFORM_VC_METHOD}", 
                headers={
                    'Content-Type' : m.content_type,
                    "Authorization": f"Bearer {self.bearer_token}",
                },
                data=m,
            )
            if audiobyte_response.status_code == 200:
                return audiobyte_response.content, "Succeeded"
            else:
                return None, eval(audiobyte_response.content)["detail"]
        except:
            return None, "Failed by Gradio Server Error"

    
    def perform_tts(
        self,
        text, 
        language, 
        model_name,
        embed,
        pitch,#=1.0 
        speed,#=1.0 
        noise_scale,#=0.1 
        sampling_rate,#=32000
        audio_clearance,#=False
    ):
        """
        Parameters
        ----------
            text:str
            language:str
            model_name:str
            pitch:float
            speed:float
            noise_scale:float
            sample_rate:int
        Returns
        -----------
            audiobyte:bytestring
            # audio:np.ndarray[T]
        """
        try:
            audiobyte_response = requests.get(
                f"{self.server_path}:{self.server_port}/{PERFORM_TTS_METHOD}", 
                json={
                    "text" : text,
                    "lang" : display_name_language[language],
                    "model_name" : self.model_display_name_dict[model_name],
                    "embed" : embed,
                    "pitch" : pitch,
                    "speed" : speed,
                    "noise_scale" : noise_scale,
                    "sampling_rate" : sampling_rate,
                    "audio_clearance" : audio_clearance,
                },
                headers = {"Authorization": f"Bearer {self.bearer_token}"},
            )
            if audiobyte_response.status_code == 200:
                # tmp_filepath = self.get_filepath(model_name, text)
                # save_audio_file_from_bytestring(audiobyte_response.content, tmp_filepath)
                return audiobyte_response.content, "Succeeded"
            else:
                return None, eval(audiobyte_response.content)["detail"]
        except:
            return None, "Failed by Gradio Server Error"