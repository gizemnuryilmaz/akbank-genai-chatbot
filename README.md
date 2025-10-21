# Coffeephil: Nitelikli Kahve Demleme Asistanı

Bu proje, Akbank GenAI Bootcamp  kapsamında geliştirilmiş, Retrieval-Augmented Generation (RAG) mimarisine sahip bir chatbot projesidir.

# 1. Projenin Amacı 

Bu projenin amacı, nitelikli kahve demleme yöntemleri hakkında bilgi veren "Kahve Gurusu" adında bir uzman chatbot oluşturmaktır. Chatbot; V60, Chemex, Aeropress gibi popüler demleme ekipmanlarının kullanımını, doğru kahve/su oranlarını, öğütme derecelerini ve su sıcaklığı gibi kritik faktörleri anlatan uzman rehberlerini veri kaynağı olarak kullanır. Kullanıcılar, "V60 için kahveyi ne kadar kalın öğütmeliyim?" gibi spesifik sorular sorarak demleme süreçlerini iyileştirebilirler.

## 2. Veri Seti Hakkında Bilgi 

Projenin bilgi tabanı (knowledge base), nitelikli kahve demleme üzerine yazılmış popüler ve güvenilir bloglardan toplanan metinlerden oluşmaktadır.

* **Toplama Metodolojisi:** V60, Chemex, Aeropress, su sıcaklığı, öğütme dereceleri ve kahve tarihi gibi spesifik konuları kapsayan 12 adet detaylı rehber metni derlenmiştir.
* **Hazırlanışı:** Her bir rehber, `.txt` formatında ayrı bir dosya olarak `data/` klasörü altında yapılandırılmıştır. Uygulama, bu dosyalardaki metinleri işleyerek kendi vektör veritabanını oluşturur.

## 3. Kullanılan Yöntemler ve Çözüm Mimarisi 
Proje, harici bilgi kullanarak dil modelinin (LLM) yeteneklerini artıran bir RAG mimarisi üzerine kuruludur. Çözüm mimarisi aşağıdaki teknolojileri içermektedir:

* **Generation Model (Üretim Modeli):** Google Gemini 2.0 Flash (Model, `google-generativeai` kütüphanesi aracılığıyla kullanılmıştır).
* **Embedding Model (Gömme Modeli):** Google `text-embedding-004` (Metinleri ve sorguları sayısal vektörlere dönüştürmek için kullanılmıştır).
* **Vektör Veritabanı:** Yerel ve bellek-içi (in-memory) bir veritabanı kullanılmıştır. Metinlerden elde edilen vektörler bir `numpy` dizisinde saklanmış ve sorgu ile karşılaştırma işlemi `sklearn.metrics.pairwise.cosine_similarity` (Kosinüs Benzerliği) ile yapılmıştır.
* **Web Arayüzü:** Projenin bir chatbot olarak sunulması için `Streamlit` kütüphanesi kullanılmıştır.

## 4. Elde Edilen Sonuçlar 

Proje başarıyla tamamlanmış ve RAG mimarisi test edilmiştir.
* Chatbot, `data/` klasöründeki bilgilere dayanarak "V60 su sıcaklığı kaç derece olmalı?" gibi spesifik sorulara doğru ve bağlamsal yanıtlar ("93-96 derece...") üretebilmektedir.
* Veri setinde bulunmayan alakasız sorulara ("Türkiye'nin başkenti neresidir?") karşı "Üzgünüm, bu kahve bilgisine sahip değilim." diyerek halüsinasyon görmesi (bilgi uydurması) başarılı bir şekilde engellenmiştir.

## 5. Yerel Ortamda Çalıştırma Kılavuzu 

Projenin yerel bilgisayarınızda çalıştırılması için aşağıdaki adımlar izlenmelidir:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/gizemnuryilmaz/akbank-genai-chatbot](https://github.com/gizemnuryilmaz/akbank-genai-chatbot)
    cd akbank-genai-chatbot
    ```

2.  **Sanal Ortam Oluşturun ve Aktifleştirin:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Ekleyin:**
    * Projenin ana dizininde `.env` adında bir dosya oluşturun.
    * İçine Google Gemini API anahtarınızı aşağıdaki gibi ekleyin:
        ```
        GEMINI_API_KEY=BURAYA_API_ANAHTARINIZI_YAPIŞTIRIN
        ```

5.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```

## 6. Web Arayüzü Linki [cite: 13]

(Bu link, proje sonunda Streamlit Cloud, Hugging Face Spaces veya benzeri bir platforma deploy edildiğinde eklenecektir.)