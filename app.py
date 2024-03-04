import tempfile

import librosa
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Streamlit 페이지 설정
st.title("Whisper Large-v3 오디오 변환")
st.write("변환할 오디오 파일을 업로드하세요.")

# 오디오 파일 업로드
audio_file = st.file_uploader(
    "오디오 파일을 선택하세요...", type=["mp3", "wav", "flac", "ogg", "mp4"]
)


@st.cache_resource(show_spinner="Loading model...")
def load_pipe():
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device}...")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 모델 ID
    model_id = "openai/whisper-large-v3"

    # 모델 및 프로세서 로드
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # 자동 음성 인식을 위한 파이프라인
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


pipe = load_pipe()

if audio_file is not None:
    # 오디오 플레이어 표시
    st.audio(audio_file, format="audio/wav", start_time=0)

    # numpy array로 바꾸기
    if audio_file is not None:
        # To read file as a byte stream
        bytes_data = audio_file.read()

        # Use librosa to load the audio file. Here, we're using a temporary file
        # workaround because librosa expects a filename.
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            # Write the byte data to a temp file
            tmp_file.write(bytes_data)
            # Seek to the start of the file
            tmp_file.seek(0)
            # Load the audio file as a numpy array
            audio, sr = librosa.load(tmp_file.name, sr=None)

    # '변환' 버튼 추가
    if st.button("오디오 변환"):
        with st.spinner("변환 중..."):
            # 오디오 파일을 파이프라인에 전달하여 변환
            result = pipe(audio)
            # 변환 결과 표시
            st.write("변환 결과:")
            st.write(result["text"])
