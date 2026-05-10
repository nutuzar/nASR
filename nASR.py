import sys
import os

# PyTorch 2.4+ ve CTranslate2 arasındaki OpenMP DLL çakışmasını (Sessiz Çökme) engellemek için kritik yama!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import subprocess
import site
import gc
import warnings
import winsound  # Donanımsal Bip sesi için

# Gereksiz kütüphane sızlanmalarını (UserWarning) gizle ki terminal temiz kalsın
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# KRİTİK YAMA: WINDOWS DLL CEHENNEMİNDEN ÇIKIŞ PROTOKOLÜ
# =====================================================================
try:
    import torch
    # PyTorch'un kendi içindeki kütüphaneleri Windows'a tanıt
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        
    # pip ile inen nvidia-cublas ve cudnn kütüphanelerini Windows'a tanıt
    for sp in site.getsitepackages():
        nvidia_cublas = os.path.join(sp, "nvidia", "cublas", "bin")
        nvidia_cudnn = os.path.join(sp, "nvidia", "cudnn", "bin")
        if os.path.exists(nvidia_cublas):
            os.add_dll_directory(nvidia_cublas)
        if os.path.exists(nvidia_cudnn):
            os.add_dll_directory(nvidia_cudnn)
except Exception as e:
    print(f"\n[SİSTEM UYARISI] DLL yaması enjekte edilirken bir pürüz çıktı / DLL patch injection error: {e}")

import whisperx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QComboBox, QListWidget, QAbstractItemView, QCheckBox,
                             QTextEdit, QPlainTextEdit, QFormLayout, QDoubleSpinBox, 
                             QSpinBox, QGroupBox, QFileDialog, QProgressBar, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor, QIcon
import re

# =====================================================================
# ÇEVİRİ SÖZLÜĞÜ (BILINGUAL DICTIONARY)
# =====================================================================
LANG = {
    "tr": {
        "title": "nASR Alt Yazı Otomasyonu - WhisperX GPU Hızlandırıcı | v1.0",
        "tab_main": "Ana İşlem Merkezi",
        "tab_adv": "Gelişmiş Ayarlar",
        "tab_about": "Hakkında",
        "queue_group": "İşlem Kuyruğu (Toplu Dosya Desteği)",
        "btn_add": "+ Dosya Ekle",
        "btn_remove": "- Seçileni Çıkar",
        "tooltip_drop": "İşlenecek medya dosyalarını buraya sürükleyip bırakabilirsiniz.\nBirden fazla dosya atarak kuyruk oluşturabilirsiniz.",
        "settings_group": "Operasyon Ayarları",
        "lbl_model": "Yapay Zeka Modeli:",
        "tooltip_model": "Base: Hızlı ama dikkatsiz.\nMedium: Hız ve doğruluk için altın oran.\nLarge-v3: 11 GB VRAM'in sınırlarını zorlar,\nkusursuz felsefi/teknik metinler için şarttır.",
        "lbl_lang": "Filmin Orijinal Dili:",
        "tooltip_lang": "Hizalama (Alignment) modellerinin sapıtmaması için\nfilmin orijinal ses dilini seçin.\nEmin değilseniz 'Otomatik' bırakın.",
        "lbl_task": "Operasyon Modu:",
        "tooltip_task": "Deşifre Et (Transcribe): Filmi kendi orijinal dilinde yazıya döker.\nİngilizce'ye Çevir (Translate): Orijinal dil ne olursa olsun,\ndoğrudan İngilizce altyazı üretir.",
        "task_transcribe": "Deşifre Et (Transcribe)",
        "task_translate": "İngilizce'ye Çevir (Translate)",
        "btn_start": "▶ KUYRUĞU BAŞLAT",
        "btn_running": "OPERASYON DEVAM EDİYOR...",
        "btn_open_folder": "📂 Çıktı Klasörünü Aç",
        "btn_corner_cancel": "🛑 İptal",
        "btn_corner_close": "❌ Kapat",
        "console_group": "Sistem Telemetrisi (Matris)",
        "lbl_telemetry_wait": "VRAM: Bekleniyor... | GPU: --°C",
        "lbl_telemetry_cpu": "GPU Telemetri: CPU Modu (Nvidia Yok)",
        "lbl_telemetry_off": "GPU Telemetri: Çevrimdışı",
        "elapsed_time": "Geçen Süre: {}",
        
        "lbl_prompt": "Özel İsim / Telaffuz İpucu\n(İsteğe Bağlı):",
        "tooltip_prompt": "Özel isimler veya telaffuzu zor kelimeler.\nModelin halüsinasyon görmesini engellemek için\nmetinde geçecek kelimeleri virgülle ayırarak yazın.\n(Örn: Heidegger, Dasein, Amor Fati)",
        "lbl_onset": "Konuşma Başlangıç Hassasiyeti\n(vad_onset):",
        "tooltip_onset": "Sesin 'konuşma' kabul edilmesi için ihtimal eşiği.\nEski ve gürültülü filmlerde bu değeri artırmak (örneğin 0.6)\nrüzgar sesinin altyazı olmasını engeller.",
        "lbl_offset": "Sessizlik Algılama Hassasiyeti\n(vad_offset):",
        "tooltip_offset": "Sesin ne zaman 'sessizlik' kabul edileceği.\nDudak kapandığında altyazının anında ekrandan\nsilinmesi için kritik bir ayardır.",
        "lbl_min_dur": "Minimum Geçerli Ses Süresi\n(min_speech_duration_ms):",
        "tooltip_min_dur": "Bu sürenin altındaki kısa tıklama, öksürme\nveya nefes alma sesleri çöpe atılır.\nSaniyenin onda biri değerler girilir.",
        "lbl_norm": "Ses Normalizasyonu ve Dip Gürültüsü Filtresi (FFmpeg)",
        "tooltip_norm": "Eski ve kısık sesli filmlerin sesini dengeler (loudnorm)\nve arka plan cızırtılarını azaltır (afftdn).",
        "lbl_filler": "Lüzumsuz Kelimeleri Ayıkla (uh, um, hmm, ah vb.)",
        "tooltip_filler": "Metin oluşturulurken duraksama ve dolgu kelimelerini\n(İngilizce) akıllıca temizler. Çeviriyi kolaylaştırır.",
        "lbl_diarize": "Konuşmacıları Ayırt Et (Diarization)",
        "tooltip_diarize": "Konuşmacıları [SPEAKER_00] şeklinde etiketler.\nÜcretsiz HuggingFace token gerektirir.",
        "tooltip_token": "Diarization için Pyannote HuggingFace Access Token'ı buraya girin.",
        "lbl_quarantine": "Düşük Güven Skoru Raporu Oluştur (Karantina Txt)",
        "tooltip_quarantine": "SRT dosyanızı işaretlerle kirletmez.\nModelin %60'ın altında emin olduğu kelimeleri\nzaman damgalarıyla birlikte harici bir TXT dosyasına yazar.",
        "lbl_json": "Hizalama Öncesi Ham Veriyi Yedekle (Termal Çökme Sigortası)",
        "tooltip_json": "Eğer VRAM şişer ve sistem çökerse,\nsaatler süren deşifre emeğinin boşa gitmemesi için\nara belleği anında diske kaydeder.",
        
        "lbl_cat_pre": "Pre-Processing:",
        "lbl_cat_denoise": "Denoising:",
        "lbl_cat_speaker": "Konuşmacı Tespiti:",
        "lbl_cat_security": "Güvenlik & Analiz:",
        "lbl_cat_crisis": "Kriz Protokolü:",
        "grp_core": "WhisperX Çekirdek Parametreleri (Dikkatli Değiştirin)",
        
        "about_error": "<h2>Hakkında dosyası ({}) bulunamadı.</h2><p>Lütfen program dizinini kontrol edin.</p>",
        "err_empty_queue": "[HATA] Kuyruk boş. Lütfen işlenecek bir film dosyası sürükleyin.",
        "msg_cage_open": "[BİLGİ] Laboratuvar kapıları kilitlendi. QThread kafesi açılıyor...",
        "log_hw_intel": "[SİSTEM] Evrensel Donanım İstihbaratı Çalıştırılıyor...",
        "log_hw_found": "[DONANIM] Nvidia GPU Bulundu:",
        "log_hw_cap": "[DONANIM] Kapasite: {:.1f} GB VRAM | Compute Capability: {}.{}",
        "log_arch_mod": "[SİSTEM] Modern GPU mimarisi algılandı. Performans için 'float16' kipi devrede.",
        "log_arch_old": "[SİSTEM] Emektar GPU (Pascal/Eski nesil) algılandı. Hayatta kalmak için 'int8' kipi devrede.",
        "log_batch": "[SİSTEM] VRAM boyutuna göre paralel işlem yükü (Batch Size) = {} olarak kilitlendi.",
        "log_no_gpu": "[UYARI] Sistemde Nvidia Ekran Kartı bulunamadı! Operasyon CPU'ya aktarılıyor.\n[KRİTİK] CPU modunda 'Large-v3' çok RAM tüketir, ağır uygulamaları kapatın!",
        "log_cancel": "\n[İPTAL] Kullanıcı acil durum frenini çekti. Operasyon acımasızca durduruldu!",
        "log_op_start": "[OPERASYON {}/{}] Başlıyor: {}",
        "log_params": "[PARAMETRELER] Model: {}, Mod: {}, Dil: {}",
        "log_stage1": "[AŞAMA 1] Ses dosyası ayrıştırılıyor (FFmpeg)...",
        "log_norm": "[SİSTEM] Ses normalizasyonu ve gürültü filtresi uygulanıyor...",
        "log_stage2": "[AŞAMA 2] {} modeli cihaz belleğine yükleniyor...",
        "log_dl_warn": "[SİSTEM] EĞER MODEL SİSTEMDE YOKSA ŞU AN SUNUCUDAN İNDİRİLİYOR DEMEKTİR.\nLütfen arayüz donmuş gibi görünse de BEKLEYİN...",
        "log_prompt": "[BİLGİ] Felsefi kalkan devrede (Initial Prompt): {}",
        "log_vram": "[BİLGİ] Operasyon başladı. Dil tespiti ve metin çıkartılıyor...",
        "log_lang_det": "[BİLGİ] Tespit edilen dil: {} - Sözlük yedeği alındı.",
        "log_json_backup": "[SİSTEM] Termal Kriz Protokolü: Ham JSON yedeği diske yazılıyor.",
        "log_stage3": "[AŞAMA 3] Hizalama (Alignment) Modeline geçiliyor...",
        "log_align_err": "[UYARI] Hizalama modelinde hata veya dil desteklenmiyor. İşlem hizalamasız devam ediyor: {}",
        "log_stage35": "[AŞAMA 3.5] Konuşmacı Ayırt Etme (Diarization) başlıyor...",
        "log_no_token": "[UYARI] HuggingFace Token yok! Diarization pas geçiliyor.",
        "log_diarize_err": "[UYARI] Diarization hatası (Token geçersiz olabilir): {}",
        "log_filler_clean": "[SİSTEM] Dolgu kelimeleri metinden ayıklanıyor...",
        "log_stage4": "[AŞAMA 4] SRT Dosyası inşa ediliyor...",
        "log_quarantine": "[BİLGİ] Karantina dosyası oluşturuluyor...",
        "log_json_clean": "[BİLGİ] Operasyon başarılı. Geçici JSON yedeği temizlendi.",
        "log_success": "[MUTLAK SONUÇ] Dosya Hazır: {}",
        "log_fatal": "\n[KATASTROFİK HATA] Sistemin karanlık köşelerinden bir hata fırladı: {}",
        "log_total_time": "\n[SİSTEM] İşlem başarılı. Toplam geçen süre: {}",
        "log_cage_close": "[SİSTEM] Kuyruktaki tüm görevler sonlandı. QThread kafesi kapatılıyor.",
        "hr_min_sec": "saat",
        "min": "dakika",
        "sec": "saniye"
    },
    "en": {
        "title": "nASR Subtitle Automation - WhisperX GPU Accelerator | v1.0",
        "tab_main": "Main Operations",
        "tab_adv": "Advanced Settings",
        "tab_about": "About",
        "queue_group": "Batch Processing Queue",
        "btn_add": "+ Add Media",
        "btn_remove": "- Remove Selected",
        "tooltip_drop": "Drag and drop media files here.\nYou can add multiple files to create a processing queue.",
        "settings_group": "Operation Settings",
        "lbl_model": "AI Engine Model:",
        "tooltip_model": "Base: Fast but lacks precision.\nMedium: The golden ratio for speed and accuracy.\nLarge-v3: Pushes the limits of 11GB VRAM,\nessential for complex/philosophical texts.",
        "lbl_lang": "Original Audio Language:",
        "tooltip_lang": "Select the original language to prevent\nalignment models from malfunctioning.\nLeave as 'Auto' if unsure.",
        "lbl_task": "Operation Mode:",
        "tooltip_task": "Transcribe: Generates subtitles in the original audio language.\nTranslate: Bypasses the original language and\ndirectly generates English subtitles.",
        "task_transcribe": "Transcribe (Original)",
        "task_translate": "Translate (To English)",
        "btn_start": "▶ START QUEUE",
        "btn_running": "OPERATION IN PROGRESS...",
        "btn_open_folder": "📂 Open Output Folder",
        "btn_corner_cancel": "🛑 Cancel",
        "btn_corner_close": "❌ Close",
        "console_group": "System Telemetry (The Matrix)",
        "lbl_telemetry_wait": "VRAM: Waiting... | GPU: --°C",
        "lbl_telemetry_cpu": "GPU Telemetry: CPU Mode (No Nvidia)",
        "lbl_telemetry_off": "GPU Telemetry: Offline",
        "elapsed_time": "Elapsed Time: {}",
        
        "lbl_prompt": "Specific Names / Prompts\n(Optional):",
        "tooltip_prompt": "Technical terms or hard-to-pronounce names.\nPrevent AI hallucinations by listing them here\nseparated by commas. (e.g. Heidegger, Dasein)",
        "lbl_onset": "Speech Onset Sensitivity\n(vad_onset):",
        "tooltip_onset": "Probability threshold for audio to be considered 'speech'.\nIncrease this (e.g., 0.6) for noisy/windy videos\nto prevent noise from becoming subtitles.",
        "lbl_offset": "Silence Detection Sensitivity\n(vad_offset):",
        "tooltip_offset": "Determines when speech ends.\nCritical setting to ensure subtitles disappear\nexactly when the speaker's mouth closes.",
        "lbl_min_dur": "Min. Valid Speech Duration\n(min_speech_duration_ms):",
        "tooltip_min_dur": "Short clicks, coughs, or breaths below this duration\nwill be trashed. Values are in milliseconds.",
        "lbl_norm": "Audio Normalization & Noise Reduction (FFmpeg)",
        "tooltip_norm": "Automatically boosts low volume (loudnorm) and\nreduces background static/noise (afftdn).",
        "lbl_filler": "Clean Filler Words (uh, um, hmm, ah)",
        "tooltip_filler": "Surgically removes hesitation and filler words\nfrom the text. Makes translation much smoother.",
        "lbl_diarize": "Speaker Diarization (Identify Speakers)",
        "tooltip_diarize": "Tags speakers as [SPEAKER_00].\nRequires a free HuggingFace access token.",
        "tooltip_token": "Paste your Pyannote HuggingFace Access Token here.",
        "lbl_quarantine": "Generate Low Confidence Report (Quarantine.txt)",
        "tooltip_quarantine": "Keeps your SRT clean. Logs words with\nless than 60% confidence and their timestamps\ninto a separate text file.",
        "lbl_json": "Backup Raw Data Pre-Alignment (Crash Insurance)",
        "tooltip_json": "If VRAM maxes out and crashes,\nsaves hours of transcription effort by dumping\nthe raw buffer to your disk.",
        
        "lbl_cat_pre": "Pre-Processing:",
        "lbl_cat_denoise": "Denoising:",
        "lbl_cat_speaker": "Speaker ID:",
        "lbl_cat_security": "Security & Analysis:",
        "lbl_cat_crisis": "Crisis Protocol:",
        "grp_core": "WhisperX Core Parameters (Modify with Caution)",
        
        "about_error": "<h2>About file ({}) not found.</h2><p>Please check the program directory.</p>",
        "err_empty_queue": "[ERROR] Queue is empty. Please drop a media file.",
        "msg_cage_open": "[INFO] Lab doors locked. Opening QThread cage...",
        "log_hw_intel": "[SYSTEM] Running Universal Hardware Intelligence...",
        "log_hw_found": "[HARDWARE] Nvidia GPU Found:",
        "log_hw_cap": "[HARDWARE] Capacity: {:.1f} GB VRAM | Compute Capability: {}.{}",
        "log_arch_mod": "[SYSTEM] Modern GPU architecture detected. 'float16' mode active.",
        "log_arch_old": "[SYSTEM] Legacy GPU (Pascal) detected. 'int8' survival mode active.",
        "log_batch": "[SYSTEM] Batch Size locked to {} based on VRAM.",
        "log_no_gpu": "[WARNING] No Nvidia GPU found! Operation routed to CPU.\n[CRITICAL] 'Large-v3' consumes massive RAM on CPU. Close heavy apps!",
        "log_cancel": "\n[CANCEL] User pulled the emergency brake. Operation aborted!",
        "log_op_start": "[OPERATION {}/{}] Starting: {}",
        "log_params": "[PARAMETERS] Model: {}, Mode: {}, Lang: {}",
        "log_stage1": "[STAGE 1] Extracting audio (FFmpeg)...",
        "log_norm": "[SYSTEM] Applying audio normalization and noise reduction...",
        "log_stage2": "[STAGE 2] Loading {} model into device memory...",
        "log_dl_warn": "[SYSTEM] IF MODEL IS MISSING, IT IS DOWNLOADING FROM SERVER RIGHT NOW.\nPlease WAIT even if the UI appears frozen...",
        "log_prompt": "[INFO] Philosophical shield active (Prompt): {}",
        "log_vram": "[INFO] Operation started. Detecting language and transcribing...",
        "log_lang_det": "[INFO] Detected language: {} - Dictionary backed up.",
        "log_json_backup": "[SYSTEM] Thermal Crisis Protocol: Writing raw JSON backup to disk.",
        "log_stage3": "[STAGE 3] Switching to Alignment Model...",
        "log_align_err": "[WARNING] Alignment error or language unsupported. Proceeding without alignment: {}",
        "log_stage35": "[STAGE 3.5] Speaker Diarization starting...",
        "log_no_token": "[WARNING] No HuggingFace Token! Skipping Diarization.",
        "log_diarize_err": "[WARNING] Diarization error (Token may be invalid): {}",
        "log_filler_clean": "[SYSTEM] Purging filler words from text...",
        "log_stage4": "[STAGE 4] Constructing SRT File...",
        "log_quarantine": "[INFO] Creating quarantine file...",
        "log_json_clean": "[INFO] Operation successful. Temporary JSON backup wiped.",
        "log_success": "[ABSOLUTE RESULT] File Ready: {}",
        "log_fatal": "\n[CATASTROPHIC ERROR] A bug crawled out of the dark: {}",
        "log_total_time": "\n[SYSTEM] Task finished successfully. Total time: {}",
        "log_cage_close": "[SYSTEM] All tasks in queue finished. QThread cage closed.",
        "hr_min_sec": "hours",
        "min": "minutes",
        "sec": "seconds"
    }
}

# =====================================================================
# KONSOL YÖNLENDİRİCİSİ: MATRİSİ ARAYÜZE AKITMAK İÇİN
# =====================================================================
class ConsoleStream(QThread):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

# =====================================================================
# QTHREAD KAFESİ: ARKA PLAN İŞÇİSİ
# =====================================================================
class WhisperWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, task_queue, params, lang="tr"):
        super().__init__()
        self.task_queue = task_queue
        self.params = params
        self.lang = lang
        self.is_running = True
        self.T = LANG[self.lang]

    def memory_flush(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        self.log_signal.emit(self.T["log_hw_intel"])
        
        try:
            total_tasks = len(self.task_queue)
            
            if torch.cuda.is_available():
                device = "cuda"
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                cap_major, cap_minor = torch.cuda.get_device_capability(0)
                
                self.log_signal.emit(f"{self.T['log_hw_found']} {torch.cuda.get_device_name(0)}")
                self.log_signal.emit(self.T["log_hw_cap"].format(vram_gb, cap_major, cap_minor))
                
                if cap_major >= 7:
                    compute_type = "float16"
                    self.log_signal.emit(self.T["log_arch_mod"])
                else:
                    compute_type = "int8"
                    self.log_signal.emit(self.T["log_arch_old"])
                    
                if vram_gb >= 10:
                    batch_size = 8
                elif vram_gb >= 6:
                    batch_size = 4
                elif vram_gb >= 4:
                    batch_size = 2
                else:
                    batch_size = 1
                
                self.log_signal.emit(self.T["log_batch"].format(batch_size))
            else:
                self.log_signal.emit(self.T["log_no_gpu"])
                device = "cpu"
                compute_type = "int8"
                batch_size = 1
            
            for index, dosya in enumerate(self.task_queue):
                if not self.is_running:
                    self.log_signal.emit(self.T["log_cancel"])
                    break
                    
                self.log_signal.emit("\n" + "═"*75)
                self.log_signal.emit(self.T["log_op_start"].format(index+1, total_tasks, os.path.basename(dosya)))
                self.log_signal.emit(self.T["log_params"].format(self.params['model'], self.params['task'], self.params['language']))
                
                # 1. Ses Hazırlığı
                self.log_signal.emit(self.T["log_stage1"])
                temp_wav = "temp_x_ses.wav"
                ffmpeg_cmd = ["ffmpeg", "-i", dosya, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
                
                if self.params.get('normalize_audio'):
                    self.log_signal.emit(self.T["log_norm"])
                    ffmpeg_cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nf=-25"])
                
                ffmpeg_cmd.extend(["-y", temp_wav])
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.progress_signal.emit(25)
                
                if not self.is_running: break
                
                # 2. Transkripsiyon (Belirsiz İlerleme Sinyali: -1)
                self.log_signal.emit(self.T["log_stage2"].format(self.params['model'].upper()))
                self.log_signal.emit(self.T["log_dl_warn"])
                
                if self.params['initial_prompt']:
                    self.log_signal.emit(self.T["log_prompt"].format(self.params['initial_prompt']))
                
                vad_options = {
                    "vad_onset": self.params['vad_onset'],
                    "vad_offset": self.params['vad_offset']
                }
                
                self.progress_signal.emit(-1) # Arayüze "Donmadım, indiriyorum" sinyali
                model = whisperx.load_model(self.params['model'], device, compute_type=compute_type, vad_options=vad_options)
                self.progress_signal.emit(35) # İndirme bitti, normale dön
                
                audio = whisperx.load_audio(temp_wav)
                
                transcribe_args = {"batch_size": batch_size, "task": self.params["task"]}
                if self.params['language'] != 'en' and self.params['language'] is not None:
                    transcribe_args['language'] = self.params['language']
                
                if self.params['initial_prompt']:
                    transcribe_args['initial_prompt'] = self.params['initial_prompt']
                
                self.log_signal.emit(self.T["log_vram"])
                result = model.transcribe(audio, **transcribe_args)
                detected_language = result["language"]
                self.log_signal.emit(self.T["log_lang_det"].format(detected_language.upper()))
                
                self.progress_signal.emit(50)
                
                del model
                self.memory_flush()
                if not self.is_running: break
                
                if self.params['json_yedek']:
                    self.log_signal.emit(self.T["log_json_backup"])
                    yedek_adi = os.path.splitext(dosya)[0] + "_yedek.json"
                    with open(yedek_adi, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 3. Hizalama
                self.log_signal.emit(self.T["log_stage3"])
                try:
                    self.progress_signal.emit(-1) # Hizalama modeli de indirilebilir
                    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
                    self.progress_signal.emit(60)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                    result["language"] = detected_language
                    del model_a
                    self.memory_flush()
                except Exception as e:
                    self.progress_signal.emit(60)
                    self.log_signal.emit(self.T["log_align_err"].format(e))
                
                self.progress_signal.emit(75)
                if not self.is_running: break
                
                # Diarization
                if self.params.get('diarize'):
                    self.log_signal.emit(self.T["log_stage35"])
                    try:
                        hf_token = self.params.get('hf_token', '')
                        if not hf_token:
                            self.log_signal.emit(self.T["log_no_token"])
                        else:
                            self.progress_signal.emit(-1)
                            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                            diarize_segments = diarize_model(audio)
                            result = whisperx.assign_word_speakers(diarize_segments, result)
                            del diarize_model
                            self.memory_flush()
                            self.progress_signal.emit(85)
                    except Exception as e:
                        self.progress_signal.emit(85)
                        self.log_signal.emit(self.T["log_diarize_err"].format(e))

                # Temizlik
                if self.params.get('clean_filler'):
                    self.log_signal.emit(self.T["log_filler_clean"])
                    filler_pattern = re.compile(r'\b(uh+|um+|hmm+|ah+|er+|uhm+)\b', re.IGNORECASE)
                    for seg in result.get("segments", []):
                        if "text" in seg:
                            cleaned = filler_pattern.sub('', seg["text"])
                            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                            seg["text"] = cleaned
                        if "words" in seg:
                            seg["words"] = [w for w in seg["words"] if not filler_pattern.match(w.get("word", "").strip(".,!?"))]

                # 4. SRT Çıktısı (Akıllı İsimlendirme)
                self.log_signal.emit(self.T["log_stage4"])
                from whisperx.utils import WriteSRT
                options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
                
                if self.params['task'] == "translate":
                    task_prefix = "EN-Ceviri"
                else:
                    task_prefix = f"{detected_language.upper()}-Orijinal"
                    
                cikti_adi = os.path.splitext(dosya)[0] + f"_{task_prefix}_nASR_{self.params['model']}.srt"
                output_dir = os.path.dirname(os.path.abspath(cikti_adi))
                if not output_dir: output_dir = "."
                
                writer = WriteSRT(output_dir)
                with open(cikti_adi, "w", encoding="utf-8") as srt:
                    writer.write_result(result, srt, options)
                
                if self.params['karantina']:
                    self.log_signal.emit(self.T["log_quarantine"])
                    karantina_dosyasi = os.path.splitext(dosya)[0] + f"_karantina.txt"
                    with open(karantina_dosyasi, "w", encoding="utf-8") as k_file:
                        k_file.write("=== KARANTİNA / QUARANTINE (<0.60 SCORE) ===\n")
                        for seg in result.get("segments", []):
                            for word in seg.get("words", []):
                                if "score" in word and word["score"] < 0.6:
                                    k_file.write(f"[{word.get('start', 0):.2f} -> {word.get('end', 0):.2f}] Kelime/Word: '{word.get('word', '')}' | Skor/Score: {word['score']:.2f}\n")
                
                if os.path.exists(temp_wav):
                    try: os.remove(temp_wav)
                    except: pass
                
                if self.params.get('json_yedek') and os.path.exists(yedek_adi):
                    try:
                        os.remove(yedek_adi)
                        self.log_signal.emit(self.T["log_json_clean"])
                    except: pass
                
                self.log_signal.emit(self.T["log_success"].format(os.path.basename(cikti_adi)))
                self.progress_signal.emit(100)
                self.msleep(1000)
                if not self.is_running: break
                self.progress_signal.emit(0) 

        except Exception as e:
            self.log_signal.emit(self.T["log_fatal"].format(str(e)))
            self.finished_signal.emit(False)
            return

        self.finished_signal.emit(True)

    def stop(self):
        self.is_running = False

# =====================================================================
# SÜRÜKLE-BIRAK DESTEKLİ LİSTE (BATCH QUEUE)
# =====================================================================
class DropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.mkv', '.mp4', '.avi', '.mp3', '.wav', '.m4a')):
                        items = [self.item(x).text() for x in range(self.count())]
                        if file_path not in items:
                            self.addItem(file_path)
        else: event.ignore()

# =====================================================================
# ANA ARAYÜZ (nASR Alt Yazı Otomasyonu)
# =====================================================================
class PascalAnvilApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = "tr"
        self.T = LANG[self.current_lang]
        self.was_canceled = False
        self.last_output_dir = ""
        
        self.setWindowTitle(self.T["title"])
        self.resize(900, 750)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: 'Segoe UI'; font-size: 10pt;}
            QTabWidget::pane { border: 1px solid #333333; border-radius: 5px; }
            QTabBar::tab { background-color: #2d2d2d; color: #888888; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background-color: #3e3e42; color: #ffffff; font-weight: bold; }
            QGroupBox { border: 1px solid #444444; border-radius: 5px; margin-top: 15px; font-weight: bold; color: #4facf7; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton:disabled { background-color: #444444; color: #888888; }
            QComboBox { background-color: #2d2d2d; border: 1px solid #444; padding: 5px; border-radius: 3px; }
            QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; border: 1px solid #444; border-radius: 3px; min-height: 25px; padding: 2px; font-weight: bold; selection-background-color: #444; selection-color: #4af626; }
            QSpinBox::up-button, QDoubleSpinBox::up-button { width: 20px; border-left: 1px solid #444; border-bottom: 1px solid #444; }
            QSpinBox::down-button, QDoubleSpinBox::down-button { width: 20px; border-left: 1px solid #444; border-top: 1px solid #444; }
            QListWidget { background-color: #121212; border: 2px dashed #444444; border-radius: 5px; padding: 5px; }
            QTextEdit, QPlainTextEdit { background-color: #0c0c0c; color: #4af626; border: 1px solid #333; font-family: 'Consolas'; }
            QLabel { color: #cccccc; }
            QToolTip { color: #ffffff; background-color: #2d2d2d; border: 1px solid #4facf7; padding: 5px; border-radius: 3px; font-size: 10pt; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Sağ üst köşedeki zarif komuta merkezi (İptal ve Kapat)
        self.corner_widget = QWidget()
        corner_layout = QHBoxLayout(self.corner_widget)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_corner_cancel = QPushButton("🛑 İptal")
        self.btn_corner_cancel.setStyleSheet("background-color: #c73636; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;")
        self.btn_corner_cancel.clicked.connect(self.cancel_process)
        self.btn_corner_cancel.hide() # Sadece işlem sırasında görünür
        
        self.btn_corner_close = QPushButton("❌ Kapat")
        self.btn_corner_close.setStyleSheet("background-color: #444444; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;")
        self.btn_corner_close.clicked.connect(self.close)
        
        corner_layout.addWidget(self.btn_corner_cancel)
        corner_layout.addWidget(self.btn_corner_close)
        
        self.tabs.setCornerWidget(self.corner_widget, Qt.Corner.TopRightCorner)

        self.init_tab_main()
        self.init_tab_advanced()
        self.init_tab_about()

        self.console_group = QGroupBox()
        console_layout = QVBoxLayout()
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMinimumHeight(120)
        console_layout.addWidget(self.console_output)
        self.console_group.setLayout(console_layout)
        main_layout.addWidget(self.console_group)

        # Butonlar Layout
        btn_action_layout = QVBoxLayout()
        
        self.btn_start = QPushButton()
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("font-size: 14pt; background-color: #007acc;")
        self.btn_start.clicked.connect(self.start_process)
        btn_action_layout.addWidget(self.btn_start)
        
        self.btn_open_folder = QPushButton()
        self.btn_open_folder.setMinimumHeight(40)
        self.btn_open_folder.setStyleSheet("font-size: 12pt; background-color: #ff9800; color: black;")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.hide()
        btn_action_layout.addWidget(self.btn_open_folder)
        
        main_layout.addLayout(btn_action_layout)

        # Telemetri ve Alt Bar
        bottom_layout = QHBoxLayout()
        
        self.lbl_heartbeat = QLabel("●")
        self.lbl_heartbeat.setStyleSheet("color: #1e1e1e; font-size: 14pt;")
        bottom_layout.addWidget(self.lbl_heartbeat)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet("QProgressBar { border: none; background-color: #333; } QProgressBar::chunk { background-color: #007acc; }")
        bottom_layout.addWidget(self.progress_bar)
        
        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color: #ff9800; font-size: 9pt; font-weight: bold; margin-left: 10px; min-width: 120px;")
        bottom_layout.addWidget(self.lbl_eta)
        
        self.lbl_telemetry = QLabel()
        self.lbl_telemetry.setStyleSheet("color: #4af626; font-size: 9pt; font-weight: bold; background-color: #111; padding: 2px 8px; border-radius: 4px; margin-left: 10px;")
        bottom_layout.addWidget(self.lbl_telemetry)
        
        signature_label = QLabel("developed by nutuzar")
        signature_label.setStyleSheet("color: #666666; font-size: 8pt; font-weight: bold; margin-left: 10px;")
        bottom_layout.addWidget(signature_label)
        
        self.btn_lang = QPushButton("🇹🇷 TR / 🇬🇧 EN")
        self.btn_lang.setFixedSize(100, 30)
        self.btn_lang.setStyleSheet("background-color: #333; color: white; padding: 2px; font-size: 9pt; margin-left: 10px;")
        self.btn_lang.clicked.connect(self.toggle_language)
        bottom_layout.addWidget(self.btn_lang)

        main_layout.addLayout(bottom_layout)

        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.update_heartbeat)
        self.heartbeat_state = False
        
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        
        # Dürüst Kronometre için yeni timer
        self.stopwatch_timer = QTimer(self)
        self.stopwatch_timer.timeout.connect(self.update_stopwatch)
        
        try:
            if torch.cuda.is_available():
                self.telemetry_timer.start(2000)
        except:
            pass

        sys.stdout = ConsoleStream()
        sys.stdout.text_written.connect(self.update_console)

        self.worker = None
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.apply_language_strings()
        self.apply_hardware_locks()

    def toggle_language(self):
        if self.current_lang == "tr":
            self.current_lang = "en"
        else:
            self.current_lang = "tr"
            
        self.T = LANG[self.current_lang]
        self.apply_language_strings()

    def apply_language_strings(self):
        self.setWindowTitle(self.T["title"])
        self.tabs.setTabText(0, self.T["tab_main"])
        self.tabs.setTabText(1, self.T["tab_adv"])
        self.tabs.setTabText(2, self.T["tab_about"])
        
        self.queue_group.setTitle(self.T["queue_group"])
        self.btn_add.setText(self.T["btn_add"])
        self.btn_remove.setText(self.T["btn_remove"])
        self.file_list.setToolTip(self.T["tooltip_drop"])
        
        self.settings_group.setTitle(self.T["settings_group"])
        self.lbl_model.setText(self.T["lbl_model"])
        self.combo_model.setToolTip(self.T["tooltip_model"])
        
        self.lbl_lang.setText(self.T["lbl_lang"])
        self.combo_lang.setToolTip(self.T["tooltip_lang"])
        if self.combo_lang.count() > 0:
            if self.current_lang == "tr":
                self.combo_lang.setItemText(0, "Otomatik")
            else:
                self.combo_lang.setItemText(0, "Auto")
            
        self.lbl_task.setText(self.T["lbl_task"])
        self.combo_task.setToolTip(self.T["tooltip_task"])
        if self.combo_task.count() > 1:
            self.combo_task.setItemText(0, self.T["task_transcribe"])
            self.combo_task.setItemText(1, self.T["task_translate"])
            
        # İptal ve Başlat butonu metinleri
        if self.worker is not None and self.worker.isRunning():
            self.btn_start.setText(self.T["btn_running"])
        else:
            self.btn_start.setText(self.T["btn_start"])
            
        self.btn_corner_cancel.setText(self.T["btn_corner_cancel"])
        self.btn_corner_close.setText(self.T["btn_corner_close"])
            
        self.btn_open_folder.setText(self.T["btn_open_folder"])
        self.console_group.setTitle(self.T["console_group"])
        
        try:
            has_cuda = torch.cuda.is_available()
        except:
            has_cuda = False
            
        if not has_cuda:
            self.lbl_telemetry.setText(self.T["lbl_telemetry_cpu"])
        else:
            if self.lbl_telemetry.text().startswith("VRAM: Bekleniyor") or self.lbl_telemetry.text().startswith("VRAM: Waiting"):
                self.lbl_telemetry.setText(self.T["lbl_telemetry_wait"])

        self.lbl_prompt.setText(self.T["lbl_prompt"])
        self.txt_prompt.setToolTip(self.T["tooltip_prompt"])
        
        self.lbl_onset.setText(self.T["lbl_onset"])
        self.spin_onset.setToolTip(self.T["tooltip_onset"])
        
        self.lbl_offset.setText(self.T["lbl_offset"])
        self.spin_offset.setToolTip(self.T["tooltip_offset"])
        
        self.lbl_min_dur.setText(self.T["lbl_min_dur"])
        self.spin_min_duration.setToolTip(self.T["tooltip_min_dur"])
        
        self.lbl_cat_pre.setText(self.T["lbl_cat_pre"])
        self.chk_normalize.setText(self.T["lbl_norm"])
        self.chk_normalize.setToolTip(self.T["tooltip_norm"])
        
        self.lbl_cat_denoise.setText(self.T["lbl_cat_denoise"])
        self.chk_clean_filler.setText(self.T["lbl_filler"])
        self.chk_clean_filler.setToolTip(self.T["tooltip_filler"])
        
        self.lbl_cat_speaker.setText(self.T["lbl_cat_speaker"])
        self.chk_diarize.setText(self.T["lbl_diarize"])
        self.chk_diarize.setToolTip(self.T["tooltip_diarize"])
        self.txt_hf_token.setToolTip(self.T["tooltip_token"])
        
        self.lbl_cat_security.setText(self.T["lbl_cat_security"])
        self.chk_quarantine.setText(self.T["lbl_quarantine"])
        self.chk_quarantine.setToolTip(self.T["tooltip_quarantine"])
        
        self.lbl_cat_crisis.setText(self.T["lbl_cat_crisis"])
        self.chk_json.setText(self.T["lbl_json"])
        self.chk_json.setToolTip(self.T["tooltip_json"])
        
        self.group_core.setTitle(self.T["grp_core"])
        
        self.load_html_content()

    def load_html_content(self):
        html_file = f"hakkinda_{self.current_lang}.html"
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), html_file)
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                self.about_text.setHtml(f.read())
        else:
            html_path_fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hakkinda.html")
            if os.path.exists(html_path_fb):
                with open(html_path_fb, "r", encoding="utf-8") as f:
                    self.about_text.setHtml(f.read())
            else:
                self.about_text.setHtml(self.T["about_error"].format(html_file))

    def on_tab_changed(self, index):
        if index == 2: 
            self.console_group.hide()
        else: 
            self.console_group.show()

    def apply_hardware_locks(self):
        self.combo_model.setCurrentText("base")
        try:
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                vram_gb = 0
        except:
            has_cuda = False
            vram_gb = 0
            
        if not has_cuda or vram_gb < 5.5:
            index = self.combo_model.findText("large-v3")
            if index >= 0:
                self.combo_model.model().item(index).setEnabled(False)
                if self.current_lang == "en":
                    msg = "Hardware Warning: 'large-v3' requires >5.5 GB VRAM."
                else:
                    msg = "Donanım Yetersiz: 'large-v3' modeli GPU ve >5.5GB VRAM gerektirir."
                self.combo_model.setItemData(index, msg, Qt.ItemDataRole.ToolTipRole)

    def update_heartbeat(self):
        self.heartbeat_state = not self.heartbeat_state
        if self.heartbeat_state:
            color = "#00ff00"
        else:
            color = "#1e1e1e"
        self.lbl_heartbeat.setStyleSheet(f"color: {color}; font-size: 14pt;")

    def update_telemetry(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu", "--format=csv,nounits,noheader"],
                encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW
            ).strip()
            parts = output.split(",")
            if len(parts) >= 3:
                vram_used = float(parts[0]) / 1024
                vram_total = float(parts[1]) / 1024
                temp = parts[2].strip()
                self.lbl_telemetry.setText(f"VRAM: {vram_used:.1f} GB / {vram_total:.1f} GB  |  GPU: {temp}°C")
        except Exception:
            self.lbl_telemetry.setText(self.T["lbl_telemetry_off"])

    def update_stopwatch(self):
        if hasattr(self, 'file_start_time') and self.worker is not None and self.worker.isRunning():
            elapsed = time.time() - self.file_start_time
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m:02d}:{s:02d}"
            self.lbl_eta.setText(self.T["elapsed_time"].format(time_str))

    def init_tab_main(self):
        self.tab_main = QWidget()
        layout = QVBoxLayout(self.tab_main)

        self.queue_group = QGroupBox()
        queue_layout = QVBoxLayout()
        self.file_list = DropListWidget()
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton()
        self.btn_add.clicked.connect(self.add_file_dialog)
        self.btn_remove = QPushButton()
        self.btn_remove.clicked.connect(self.remove_selected_file)
        self.btn_remove.setStyleSheet("background-color: #c73636;")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        
        queue_layout.addWidget(self.file_list)
        queue_layout.addLayout(btn_layout)
        self.queue_group.setLayout(queue_layout)
        layout.addWidget(self.queue_group)

        self.settings_group = QGroupBox()
        settings_layout = QHBoxLayout()

        model_layout = QVBoxLayout()
        self.lbl_model = QLabel()
        self.combo_model = QComboBox()
        self.combo_model.addItems(["base", "medium", "large-v3"])
        self.combo_model.setCurrentText("base")
        model_layout.addWidget(self.lbl_model)
        model_layout.addWidget(self.combo_model)
        settings_layout.addLayout(model_layout)

        lang_layout = QVBoxLayout()
        self.lbl_lang = QLabel()
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["Otomatik", "en", "fr", "de", "es", "tr"])
        lang_layout.addWidget(self.lbl_lang)
        lang_layout.addWidget(self.combo_lang)
        settings_layout.addLayout(lang_layout)

        task_layout = QVBoxLayout()
        self.lbl_task = QLabel()
        self.combo_task = QComboBox()
        self.combo_task.addItems(["Deşifre Et (Transcribe)", "İngilizce'ye Çevir (Translate)"])
        task_layout.addWidget(self.lbl_task)
        task_layout.addWidget(self.combo_task)
        settings_layout.addLayout(task_layout)

        self.settings_group.setLayout(settings_layout)
        layout.addWidget(self.settings_group)
        self.tabs.addTab(self.tab_main, "")

    def init_tab_advanced(self):
        self.tab_adv = QWidget()
        layout = QVBoxLayout(self.tab_adv)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        self.form_layout = QFormLayout()
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_layout.setSpacing(15)

        self.lbl_prompt = QLabel()
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setMaximumHeight(45)
        self.txt_prompt.setPlaceholderText("Heidegger, Dasein, Amor Fati...")
        self.form_layout.addRow(self.lbl_prompt, self.txt_prompt)

        self.lbl_onset = QLabel()
        self.spin_onset = QDoubleSpinBox()
        self.spin_onset.setRange(0.01, 1.0)
        self.spin_onset.setSingleStep(0.05)
        self.spin_onset.setValue(0.500)
        self.form_layout.addRow(self.lbl_onset, self.spin_onset)

        self.lbl_offset = QLabel()
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(0.01, 1.0)
        self.spin_offset.setSingleStep(0.05)
        self.spin_offset.setValue(0.363)
        self.form_layout.addRow(self.lbl_offset, self.spin_offset)

        self.lbl_min_dur = QLabel()
        self.spin_min_duration = QSpinBox()
        self.spin_min_duration.setRange(0, 5000)
        self.spin_min_duration.setSingleStep(50)
        self.spin_min_duration.setValue(250)
        self.spin_min_duration.setSuffix(" ms")
        self.form_layout.addRow(self.lbl_min_dur, self.spin_min_duration)

        self.lbl_cat_pre = QLabel()
        self.chk_normalize = QCheckBox()
        self.chk_normalize.setChecked(False)
        self.form_layout.addRow(self.lbl_cat_pre, self.chk_normalize)

        self.lbl_cat_denoise = QLabel()
        self.chk_clean_filler = QCheckBox()
        self.chk_clean_filler.setChecked(False)
        self.form_layout.addRow(self.lbl_cat_denoise, self.chk_clean_filler)
        
        self.lbl_cat_speaker = QLabel()
        self.chk_diarize = QCheckBox()
        self.chk_diarize.setChecked(False)
        self.txt_hf_token = QTextEdit()
        self.txt_hf_token.setMaximumHeight(30)
        self.txt_hf_token.setPlaceholderText("hf_...")
        diarize_layout = QVBoxLayout()
        diarize_layout.addWidget(self.chk_diarize)
        diarize_layout.addWidget(self.txt_hf_token)
        self.form_layout.addRow(self.lbl_cat_speaker, diarize_layout)

        self.lbl_cat_security = QLabel()
        self.chk_quarantine = QCheckBox()
        self.chk_quarantine.setChecked(True)
        self.form_layout.addRow(self.lbl_cat_security, self.chk_quarantine)

        self.lbl_cat_crisis = QLabel()
        self.chk_json = QCheckBox()
        self.chk_json.setChecked(True)
        self.form_layout.addRow(self.lbl_cat_crisis, self.chk_json)

        self.group_core = QGroupBox()
        self.group_core.setLayout(self.form_layout)
        scroll_layout.addWidget(self.group_core)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self.tabs.addTab(self.tab_adv, "")

    def init_tab_about(self):
        self.tab_about = QWidget()
        layout = QVBoxLayout(self.tab_about)
        self.about_text = QTextEdit()
        self.about_text.setReadOnly(True)
        self.about_text.setStyleSheet("background-color: #121212; border: none; padding: 10px;")
        layout.addWidget(self.about_text)
        self.tabs.addTab(self.tab_about, "")

    def add_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Media", "", "Medya (*.mkv *.mp4 *.avi *.wav *.mp3 *.m4a)")
        for f in files:
            items = []
            for x in range(self.file_list.count()):
                items.append(self.file_list.item(x).text())
            if f not in items: 
                self.file_list.addItem(f)

    def remove_selected_file(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    @pyqtSlot(str)
    def update_console(self, text):
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)
        self.console_output.insertPlainText(text)
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)

    @pyqtSlot(str)
    def update_log(self, text):
        import time as t_mod
        now = t_mod.strftime("%H:%M:%S")
        if text.strip() and not text.strip().startswith("═"):
            if text.startswith("\n"):
                text = f"\n[{now}] {text[1:]}" 
            else:
                text = f"[{now}] {text}"
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)
        self.console_output.insertPlainText(text + "\n")
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)
        
    @pyqtSlot(int)
    def handle_progress(self, val):
        if val == -1:
            self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(val)
            
            if val == 0:
                self.file_start_time = time.time()
                self.lbl_eta.setText(self.T["elapsed_time"].format("00:00"))

    def open_output_folder(self):
        if os.path.exists(self.last_output_dir):
            os.startfile(self.last_output_dir)

    def cancel_process(self):
        if self.worker is not None and self.worker.isRunning():
            self.was_canceled = True
            self.worker.stop()
            self.btn_start.setText(self.T["btn_running"])
            self.update_log("\n[SİSTEM] Kullanıcıdan İPTAL emri alındı. Kafes zorla kapatılıyor...")
            try:
                winsound.Beep(1500, 300)
                winsound.Beep(1000, 500)
            except:
                pass

    def start_process(self):
        if self.file_list.count() == 0:
            print(self.T["err_empty_queue"])
            return

        self.was_canceled = False
        self.btn_open_folder.hide()
        
        ilk_dosya = self.file_list.item(0).text()
        self.last_output_dir = os.path.dirname(os.path.abspath(ilk_dosya))

        task_text = self.combo_task.currentText()
        if "Translate" in task_text or "Çevir" in task_text:
            task_mode = "translate"
        else:
            task_mode = "transcribe"
            
        lang_text = self.combo_lang.currentText()
        if "Auto" in lang_text or "Otomatik" in lang_text:
            lang_code = None
        else:
            lang_code = lang_text.split(" ")[0]

        params = {
            "model": self.combo_model.currentText(),
            "language": lang_code,
            "task": task_mode,
            "initial_prompt": self.txt_prompt.toPlainText().strip(),
            "vad_onset": self.spin_onset.value(),
            "vad_offset": self.spin_offset.value(),
            "min_speech_duration_ms": self.spin_min_duration.value(),
            "karantina": self.chk_quarantine.isChecked(),
            "json_yedek": self.chk_json.isChecked(),
            "normalize_audio": self.chk_normalize.isChecked(),
            "clean_filler": self.chk_clean_filler.isChecked(),
            "diarize": self.chk_diarize.isChecked(),
            "hf_token": self.txt_hf_token.toPlainText().strip()
        }

        queue = []
        for x in range(self.file_list.count()):
            queue.append(self.file_list.item(x).text())

        # Ana buton artık sadece "DURUM" bildirir, tıklanamaz
        self.btn_start.setEnabled(False)
        self.btn_start.setText(self.T["btn_running"])
        self.btn_start.setStyleSheet("font-size: 14pt; background-color: #888888; color: #444;")
        
        # Köşedeki iptal butonu aktifleşir
        self.btn_corner_cancel.show()
        
        self.heartbeat_timer.start(500)
        self.file_list.setEnabled(False)
        self.btn_lang.setEnabled(False)
        self.tabs.setTabEnabled(1, False)

        print(f"\n{self.T['msg_cage_open']}")
        
        self.start_time = time.time()
        self.file_start_time = time.time()
        self.stopwatch_timer.start(1000) # Saniyede bir kronometreyi güncelle
        
        self.worker = WhisperWorker(queue, params, self.current_lang)
        self.worker.log_signal.connect(self.update_log)
        self.worker.progress_signal.connect(self.handle_progress)
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.start()

    def process_finished(self, success):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.stopwatch_timer.stop()
        self.lbl_eta.setText("")
        self.heartbeat_timer.stop()
        self.lbl_heartbeat.setStyleSheet("color: #1e1e1e; font-size: 14pt;")
        
        self.btn_corner_cancel.hide()
        
        if hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            
            str_hr = self.T["hr_min_sec"]
            str_min = self.T["min"]
            str_sec = self.T["sec"]
            
            if hours > 0:
                time_str = f"{hours} {str_hr} {mins} {str_min} {secs} {str_sec}"
            else:
                time_str = f"{mins} {str_min} {secs} {str_sec}"
                
            self.update_log(self.T["log_total_time"].format(time_str))
            
        self.update_log(self.T["log_cage_close"])
        
        # Ana butonu tekrar aktif et
        self.btn_start.setText(self.T["btn_start"])
        self.btn_start.setStyleSheet("font-size: 14pt; background-color: #007acc; color: white;")
        self.btn_start.setEnabled(True)
        
        self.file_list.setEnabled(True)
        self.btn_lang.setEnabled(True)
        self.tabs.setTabEnabled(1, True)
        
        if success and not self.was_canceled:
            self.btn_open_folder.show()
            try:
                winsound.Beep(1000, 1000) # 1 Saniyelik net bir bip sesi
            except:
                pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = PascalAnvilApp()
    window.show()
    sys.exit(app.exec())