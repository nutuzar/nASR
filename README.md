# 🎬 nASR: Autonomous Subtitle Generation Engine (v1.0)
*AI-powered, portable, and fully autonomous WhisperX subtitle generation tool with zero internet dependency.*

---

## 🌍 English 

nASR is a highly optimized, portable automation engine built on **WhisperX**. It provides a fully autonomous workflow for transcribing and translating audio/video files. It operates entirely on your local hardware, requiring an internet connection **only once** to download the necessary AI models.

### 📥 Download the Portable Package
Since the autonomous Python environment and CUDA libraries required to run the AI on your GPU locally exceed GitHub's file size limits, the actual executable package is hosted separately.

👉 **[DOWNLOAD nASR v1.0 PORTABLE PACKAGE (3 GB) HERE](https://drive.google.com/file/d/1edcL4HlvCoYeX5QT-Elwiz3G8mnFJ1Jb/view?usp=drive_link)**

*(Extract the `.zip` file and simply run `Baslat.bat` to launch the automation.)*

### ⚙️ Core Failsafes & Features
- **Universal Hardware Intelligence:** Autonomously detects your hardware. If VRAM is below 5.5GB or no Nvidia GPU is found, it automatically disables the heavy `large-v3` model and seamlessly routes operations to the CPU to prevent system crashes.
- **Thermal Crash Insurance:** Dumps the raw transcription data to your disk as a JSON backup right before the alignment phase. If your GPU hits thermal limits and crashes, your transcription effort of hours won't go to waste.
- **Quarantine Report:** Doesn't pollute your `.srt` file. Words with less than a 60% confidence score are logged into a separate `_karantina.txt` file along with their timestamps.
- **Smart Diarization & Denoising:** Identifies speakers (requires Pyannote HF Token), normalizes audio, filters background static (`afftdn`), and surgically removes filler words (uh, um, hmm) for flawless translation preparation.
- **Bilingual Interface:** The UI fully supports both English and Turkish, switchable in real-time.

---

## 🇹🇷 Türkçe

nASR, sesten metne dönüşümde dünyanın en hızlı motoru olan **WhisperX** altyapısı üzerine inşa edilmiş; tamamen taşınabilir, otonom ve yerel donanımda çalışan bir altyazı otomasyonudur. Program gerekli yapay zeka modellerini sadece ilk kullanımda indirir ve sonrasında operasyonlar sırasında **kesinlikle internet bağlantısı gerektirmez.**

### 📥 Taşınabilir Paketi İndirin
Yapay zekanın ekran kartınızda (GPU) otonom çalışması için gereken yalıtılmış Python ortamı ve devasa CUDA kütüphaneleri GitHub'ın dosya boyutu sınırlarını aştığı için, çalışan asıl paket harici olarak sunulmaktadır.

👉 **[nASR v1.0 TAŞINABİLİR PAKETİNİ (3 GB) BURADAN İNDİRİN](https://drive.google.com/file/d/1edcL4HlvCoYeX5QT-Elwiz3G8mnFJ1Jb/view?usp=drive_link)**

*(İndirdiğiniz `.zip` dosyasını klasöre çıkartın ve programı çalıştırmak için `Baslat.bat` dosyasına tıklayın.)*

### ⚙️ Güvenlik Sigortaları ve Temel Özellikler
- **Evrensel Donanım İstihbaratı:** Sistem açılışında donanımınızı milimetrik olarak tarar. VRAM'iniz 5.5 GB'ın altındaysa veya Nvidia GPU yoksa, sistem çökmesini engellemek için ağır `large-v3` modelini otomatik kilitler ve işlemi CPU'ya devreder.
- **Termal Çökme Sigortası (JSON Yedeği):** Ekran kartınız sınırlarına dayanıp çökerse, saatler süren emeğiniz boşa gitmez. Program, hizalama (alignment) aşamasına geçmeden saniyeler önce ham metni diskinize otonom olarak yedekler.
- **Karantina Raporu:** Altyazı dosyanızı kirletmez. Modelin %60'ın altında bir güvenle yazdığı şüpheli kelimeleri, zaman damgalarıyla birlikte harici bir `_karantina.txt` dosyasına raporlar. 
- **Akıllı Denoising ve Konuşmacı Tespiti:** Sesi stüdyo standartlarına eşitler, dip gürültüsünü filtreler ve metindeki lüzumsuz dolgu kelimelerini (uh, um, hmm) kazıyarak atar. İstenirse Pyannote Token ile konuşmacıları ayırt edebilir.

---

> *"nASR, bilginin ve yazılımın özgürce paylaşılması, geliştirilmesi ve elitist duvarlar ardına hapsedilmemesi gerektiğine olan sarsılmaz inancımın bir ürünüdür. Bu proje, açık kaynak felsefesinin kolektif gücüne duyduğum saygının dijital bir tezahürüdür."* — **nutuzar**

**Developer:** nutuzar
**Version:** 1.0 (Stable)
**Contact:** nutuzar@gmail.com
