"""
Streamlit web app for voice similarity search
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from src.search.similarity_search import VoiceSimilaritySearch
from src.utils.audio_utils import save_audio
import os

# Page config
st.set_page_config(
    page_title="Voice Similarity Search",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .similarity-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_system' not in st.session_state:
    try:
        st.session_state.search_system = VoiceSimilaritySearch()
        st.session_state.system_ready = True
    except Exception as e:
        st.session_state.system_ready = False
        st.session_state.error_message = str(e)

# Header
st.markdown('<div class="main-header">🎤 Hệ thống Tìm kiếm Giọng nói Phụ nữ</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - System Info
with st.sidebar:
    st.header("📊 Thông tin Hệ thống")
    
    if st.session_state.system_ready:
        stats = st.session_state.search_system.get_system_stats()
        st.metric("Số lượng giọng nói", stats.get('total_vectors', 0))
        st.metric("Số chiều đặc trưng", stats.get('feature_dimension', 0))
        st.info(f"**Database:** {stats.get('index_type', 'N/A')}")
    else:
        st.error("⚠️ Hệ thống chưa sẵn sàng")
        st.write(st.session_state.get('error_message', 'Vui lòng build FAISS index trước'))
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Hướng dẫn
    1. Tải lên file âm thanh giọng phụ nữ
    2. Chờ hệ thống xử lý
    3. Xem kết quả tìm kiếm top 5 giọng tương đồng
    
    **Định dạng:** WAV, MP3, FLAC
    **Thời lượng:** 3-10 giây (tối ưu)
    """)

# Main content
if not st.session_state.system_ready:
    st.error("🚫 Hệ thống chưa sẵn sàng. Vui lòng build FAISS index trước khi sử dụng.")
    st.code("""
    # Chạy các lệnh sau để setup:
    python src/data_collection/download_audio.py
    python src/data_collection/preprocess_audio.py
    python scripts/build_database.py
    """)
    st.stop()

# File upload
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Tải lên âm thanh tìm kiếm")
    uploaded_file = st.file_uploader(
        "Chọn file âm thanh giọng phụ nữ",
        type=['wav', 'mp3', 'flac'],
        help="Tải lên file âm thanh để tìm các giọng nói tương đồng"
    )

with col2:
    st.subheader("⚙️ Tùy chọn")
    top_k = st.slider("Số kết quả", min_value=1, max_value=10, value=5)
    show_waveform = st.checkbox("Hiển thị dạng sóng", value=True)

# Process uploaded file
if uploaded_file is not None:
    st.markdown("---")
    
    # Save uploaded file temporarily
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / "query_audio.wav"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Display query audio
    st.subheader("🎵 Âm thanh đầu vào")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.audio(str(temp_file_path), format='audio/wav')
    
    with col_b:
        if show_waveform:
            # Plot waveform
            y, sr = librosa.load(str(temp_file_path))
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4')
            ax.set_title("Dạng sóng âm thanh đầu vào")
            ax.set_xlabel("Thời gian (s)")
            ax.set_ylabel("Biên độ")
            st.pyplot(fig)
            plt.close()
    
    # Search similar voices
    st.subheader("🔍 Kết quả tìm kiếm")
    
    with st.spinner('Đang phân tích và tìm kiếm giọng nói tương đồng...'):
        try:
            results = st.session_state.search_system.search_similar(
                str(temp_file_path),
                top_k=top_k
            )
            
            # Display results
            if results:
                for rank, (file_path, similarity, distance) in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>#{rank} - Kết quả</h3>
                        <p class="similarity-score">Độ tương đồng: {similarity:.1f}%</p>
                        <p><strong>File:</strong> {Path(file_path).name}</p>
                        <p><strong>Khoảng cách L2:</strong> {distance:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio player
                    if os.path.exists(file_path):
                        col_c, col_d = st.columns([1, 1])
                        
                        with col_c:
                            st.audio(file_path, format='audio/wav')
                        
                        with col_d:
                            if show_waveform:
                                # Plot waveform
                                y_result, sr_result = librosa.load(file_path)
                                fig_result, ax_result = plt.subplots(figsize=(8, 3))
                                librosa.display.waveshow(y_result, sr=sr_result, ax=ax_result, color='#2ca02c')
                                ax_result.set_title(f"Dạng sóng #{rank}")
                                ax_result.set_xlabel("Thời gian (s)")
                                ax_result.set_ylabel("Biên độ")
                                st.pyplot(fig_result)
                                plt.close()
                    else:
                        st.warning(f"⚠️ File không tìm thấy: {file_path}")
                    
                    st.markdown("---")
            else:
                st.info("Không tìm thấy kết quả phù hợp")
                
        except Exception as e:
            st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")
            st.exception(e)
    
    # Cleanup
    if temp_file_path.exists():
        temp_file_path.unlink()

else:
    # Show sample info when no file uploaded
    st.info("👆 Vui lòng tải lên file âm thanh để bắt đầu tìm kiếm")
    
    # Sample stats
    if st.session_state.system_ready:
        st.markdown("### 📈 Thống kê Database")
        stats = st.session_state.search_system.get_system_stats()
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Tổng số giọng", stats.get('total_vectors', 0))
        with col_stats2:
            st.metric("Chiều đặc trưng", stats.get('feature_dimension', 0))
        with col_stats3:
            st.metric("Loại index", stats.get('index_type', 'N/A'))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Hệ thống Tìm kiếm Giọng nói dựa trên Độ tương đồng | Powered by FAISS & Librosa</p>
</div>
""", unsafe_allow_html=True)
