import os
import requests
import gradio as gr
import numpy as np

from src.request_utils import RequestManager

SERVER_PATH = os.environ.get("SERVER_PATH") # Ex) "http://192.168.0.17"
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
AUDIOPATH = os.environ.get("AUDIOPATH")

SERVER_PORT = int(os.environ.get("SERVER_PORT", 8000))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", 6666))
ID = os.environ.get("ID", "xinapse")
PASSWORD = os.environ.get("PASSWORD", "1234")

request_manager = RequestManager(SERVER_PATH, SERVER_PORT, BEARER_TOKEN, AUDIOPATH)

tts_model_name_list_in_server = request_manager.get_server_tts_model_list()
vc_model_name_list_in_server = request_manager.get_server_vc_model_list()

is_tts_model_loaded = len(tts_model_name_list_in_server) > 0
is_vc_model_loaded = len(vc_model_name_list_in_server) > 0


if is_tts_model_loaded:
    tts_initial_embed_list = request_manager.get_server_embed_list(tts_model_name_list_in_server[0])
    tts_initial_language_list = request_manager.get_server_language_list(tts_model_name_list_in_server[0])
else:
    tts_model_name_list_in_server = ["없음"]
    tts_initial_embed_list = ["없음"]
    tts_initial_language_list = ["없음"]

if is_vc_model_loaded:
    vc_initial_embed_list = request_manager.get_server_embed_list(vc_model_name_list_in_server[0])
else:
    vc_model_name_list_in_server = ["없음"]
    vc_initial_embed_list = ["없음"]

sampling_rate_list = [48000, 44100, 32000, 24000, 16000, 8000]

def change_embed_dropdown(model_name):
    embed_list = request_manager.get_server_embed_list(model_name)
    return gr.Dropdown(
        choices=embed_list,
        label="스타일",
        value=embed_list[0],
    )

def change_language_dropdown(model_name):
    language_list = request_manager.get_server_language_list(model_name)
    return gr.Dropdown(
        choices=language_list,
        label="언어",
        value=language_list[0],
    )

def change_download_button(model_name, text, output_audio):
    if output_audio is None:
        return gr.DownloadButton(
            value=None,
            visible=False,
            label="다운로드",
        )
    else:
        return gr.DownloadButton(
            value=request_manager.save_audio_file(model_name, text, output_audio),
            visible=True,
            label="다운로드",
        )

def audio_input_toggle(choice):
    if choice == "mic":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(css="자이냅스 CPU TTS", theme='sudeepshouche/minimalist') as app:
    gr.Markdown(
        value='''
# 자이냅스 CPU 음성 AI Web UI 페이지    
## 사용법
   - 사용하실 모델의 모델 명을 선택하시면 해당 모델에서 사용가능한 스타일, TTS의 경우 언어 목록이 갱신됩니다.
   - 언어의 경우, 모델마다 구사할 수 있는 언어 목록이 제한되어 있으며, 생성하시려는 텍스트의 언어를 선택하여 주십시오.
   - 스타일의 경우 모델마다 기본 한개씩 제공되나, 감정과 같이 다양한 스타일이 제공될 경우, 선택하여 원하시는 스타일의 음성을 만들 수 있습니다.
   - TTS의 경우, 텍스트를 작성해주시고, 생성 버튼을 누르시면 음성이 생성됩니다.
   - Voice Conversion의 경우, 변조할 음성을 업로드 해주시고, 생성 버튼을 누르시면 음성이 생성됩니다.
   - 음성의 톤을 높낮이 파라미터를 사용하여 조절할 수 있습니다. 파라미터가 1보다 커지면 톤이 높아지고, 1보다 작아지면 톤이 낮아집니다.
   - TTS의 경우 음성의 말하는 속도를 길이 파라미터를 사용하여 조절할 수 있습니다. 파라미터가 1보다 커지면 느려지고, 1보다 작아지면 빨라집니다.
   - 오디오 후처리를 통해 음성에 포함된 노이즈 일부를 삭제할 수 있습니다. 
   - 생성된 음성의 오른쪽 위 다운로드 버튼으로 음성을 다운로드 할 수 있습니다.''',
   )
    
    with gr.Row(equal_height=True):
        ###### 음성 합성을 위한 UI Tab
        with gr.Tab("음성 합성 TTS", visible=is_tts_model_loaded):
            with gr.Column(elem_classes=['container'], scale=5):
                tts_text = gr.Textbox(
                    value="안녕하세요 자이냅스의 TTS 입니다.", 
                    label="텍스트",
                    lines=8,
                    info="텍스트를 입력하십시오",
                )
                tts_generate_button = gr.Button("생성")
    
                tts_output_audio = gr.Audio(
                    label="생성된 음성",
                    autoplay=True,
                    show_download_button=False,
                    interactive=False,
                    format="wav",
                )
    
                tts_download_button = gr.DownloadButton(
                    label="다운로드",
                    interactive=True,
                    visible=False,
                )
            
                tts_output_text = gr.Textbox(
                    label="상태창",
                    interactive=False,
                    value="Waiting",
                )
    
            with gr.Column(elem_classes=['container'], scale=2):
                tts_model_name = gr.Dropdown(
                    choices=tts_model_name_list_in_server,
                    label="모델",
                    value=tts_model_name_list_in_server[0],
                )
    
                tts_language = gr.Dropdown(
                    choices=tts_initial_language_list,
                    interactive=True, 
                    label="언어",
                    value=tts_initial_language_list[0],
                )
        
                tts_embed = gr.Dropdown(
                    choices=tts_initial_embed_list, 
                    interactive=True, 
                    label="스타일",
                    value=tts_initial_embed_list[0],
                )
    
                tts_audio_clearance = gr.Checkbox(
                    value=False,
                    label="오디오 후처리 여부",
                )
            
                tts_pitch = gr.Slider(
                    minimum=0.7, 
                    maximum=1.3, 
                    value=1.0, 
                    label="높낮이"
                )
            
                tts_speed = gr.Slider(
                    minimum=0.7, 
                    maximum=1.3, 
                    value=1.0, 
                    label="길이",
                )
            
                tts_noise_scale = gr.Slider(
                    minimum=0., 
                    maximum=1., 
                    value=0.1, 
                    label="노이즈 스케일"
                )
            
                tts_sampling_rate = gr.Dropdown(
                    choices=sampling_rate_list,
                    value=32000,
                    label="샘플링 레이트",
                )

        ###### 음성 변조를 위한 UI Tab
        with gr.Tab("음성 변조 Voice Conversion", visible=is_vc_model_loaded):
            with gr.Column(elem_classes=['container'], scale=5):
                vc_filename_text = gr.Textbox(
                    label="파일명",
                    interactive=False,
                    visible=False,
                    value="voice_conversion",
                )
                
                vc_input_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    # interactive=True,
                    label="입력 음성",
                )
                vc_generate_button = gr.Button("생성")
    
                vc_output_audio = gr.Audio(
                    label="생성된 음성",
                    autoplay=True,
                    show_download_button=False,
                    interactive=False,
                    format="wav",
                )
    
                vc_download_button = gr.DownloadButton(
                    label="다운로드",
                    interactive=True,
                    visible=False,
                )
            
                vc_output_text = gr.Textbox(
                    label="상태창",
                    interactive=False,
                    value="Waiting",
                )
    
            with gr.Column(elem_classes=['container'], scale=2):
                vc_model_name = gr.Dropdown(
                    choices=vc_model_name_list_in_server,
                    label="모델",
                    value=vc_model_name_list_in_server[0],
                )
        
                vc_embed = gr.Dropdown(
                    choices=vc_initial_embed_list, 
                    interactive=True, 
                    label="스타일",
                    value=vc_initial_embed_list[0],
                )
    
                vc_audio_clearance = gr.Checkbox(
                    value=False,
                    label="오디오 후처리 여부",
                )
            
                vc_pitch = gr.Slider(
                    minimum=0.7, 
                    maximum=1.3, 
                    value=1.0, 
                    label="높낮이"
                )
            
                vc_noise_scale = gr.Slider(
                    minimum=0., 
                    maximum=1., 
                    value=0.1, 
                    label="노이즈 스케일"
                )
            
                vc_sampling_rate = gr.Dropdown(
                    choices=sampling_rate_list,
                    value=32000,
                    label="샘플링 레이트",
                )

    ###### Xinapse Logo Image
    gr.HTML("<img src='file/assets/XinapseLogo.png'>")

    ###### TTS Interactives
    tts_model_name.change(change_language_dropdown, inputs=tts_model_name, outputs=tts_language)
    tts_model_name.change(change_embed_dropdown, inputs=tts_model_name, outputs=tts_embed)
    tts_output_audio.change(change_download_button, inputs=[tts_model_name, tts_text, tts_output_audio], outputs=tts_download_button)
    tts_generate_button.click(
        fn=request_manager.perform_tts, 
        inputs=[
            tts_text, 
            tts_language,
            tts_model_name,
            tts_embed,
            tts_pitch,
            tts_speed,
            tts_noise_scale,
            tts_sampling_rate,
            tts_audio_clearance,
        ],
        outputs=[tts_output_audio, tts_output_text]
    )

    ###### Voice Conversion Interactives
    vc_model_name.change(change_embed_dropdown, inputs=vc_model_name, outputs=vc_embed)
    vc_output_audio.change(change_download_button, inputs=[vc_model_name, vc_filename_text, vc_output_audio], outputs=vc_download_button)
    vc_generate_button.click(
        fn=request_manager.perform_vc, 
        inputs=[
            vc_input_audio, 
            vc_model_name,
            vc_embed,
            vc_pitch,
            vc_noise_scale,
            vc_sampling_rate,
            vc_audio_clearance,
        ],
        outputs=[vc_output_audio, vc_output_text]
    )

####### Gradio App Launch
### ssl_certificate : 
# 출처 : https://github.com/gradio-app/gradio/issues/2551, when audio microphone for record not found!
app.queue()
app.launch(
    auth=(ID, PASSWORD),
    auth_message="아이디와 패스워드를 입력하세요",
    server_port=GRADIO_PORT,
    share=False, 
    server_name="0.0.0.0", 
    allowed_paths=["."],
    # ssl_certfile="./certificate/cert.pem", 
    # ssl_keyfile="./certificate/key.pem",
    # ssl_verify=False,
)