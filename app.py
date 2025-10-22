import streamlit as st
import os
import glob
import numpy as np
import pandas as pd # <-- YENİ: Pandas kütüphanesini ekliyoruz
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import time # <-- Yeni import
import chromadb # <-- YENİ İMPORT
import requests
import json

# --- RAG İÇİN ÖZEL CSV İŞLEME FONKSİYONU ---

def create_rag_document_from_row(row: pd.Series) -> str:
    """
    Pandas Series (tek bir CSV satırı) objesini RAG için anlamlı bir string'e dönüştürür.
    
    Bu format, daha önceki yanıtlarda belirlenen sütun başlıklarına uygundur.
    """
    
    # NaN veya boş değerleri yönetmek için yardımcı fonksiyon
    def safe_get(key):
        # row.get(key, '') ile sütun bulunamazsa veya değeri boşsa/NaN ise None döndür
        value = str(row.get(key, '')).strip()
        if value.lower() in ('nan', '', '-') or key not in row:
            return None
        return value

    # Veri setinizdeki sütun başlıklarını kullanarak değerleri al
    name = safe_get('name')
    roaster = safe_get('roaster')
    roast = safe_get('roast')
    rating = safe_get('rating')
    origin_1 = safe_get('origin_1')
    origin_2 = safe_get('origin_2')
    price = safe_get('100g_USD')
    desc_1 = safe_get('desc_1')
    desc_2 = safe_get('desc_2')
    desc_3 = safe_get('desc_3')
    review_date = safe_get('review_date')

    parts = []

    # 1. Tanımlayıcı Bilgiler
    if name and roaster:
        parts.append(f"Kahve Adı: {name} (Kavurucu: {roaster}).")
    
    # 2. Kökenler
    origins = [o for o in [origin_1, origin_2] if o]
    if origins:
        parts.append(f"Köken Ülke(ler): {', '.join(origins)}.")
    
    # 3. Kavurma, Puan ve Fiyat
    if roast:
        parts.append(f"Kavurma Seviyesi: {roast}.")
    if rating:
        parts.append(f"Puan: {rating}.")
    if price:
        parts.append(f"Fiyatı: {price} USD/100g.")
    
    # 4. Açıklamalar (Tadım Notları)
    descriptions = [d for d in [desc_1, desc_2, desc_3] if d]
    if descriptions:
        parts.append(f"Tadım Notları: {', '.join(descriptions)}.")
        
    # 5. Ekstra Bilgi
    if review_date:
        parts.append(f"İnceleme Tarihi: {review_date}.")

    # Tüm parçaları tek bir string belge olarak birleştir
    return " ".join(parts)


# --- 1. YÜKLEME VE KURULUM (GÜNCELLENMİŞ) ---

# .env dosyasındaki API anahtarını yükle
load_dotenv() 
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

try:
    # API anahtarını ortam değişkeninden alarak ayarla
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Hata: GOOGLE_API_KEY bulunamadı veya boş. Lütfen .env dosyanızı kontrol edin.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"API anahtarı ayarlanırken bir hata oluştu: {e}")
    st.stop()

# Embedding modeli
EMBEDDING_MODEL = 'gemini-embedding-001'



BATCH_SIZE = 4000  # Kota aşımını yönetmek için
SLEEP_TIME = 1  # saniye (Kota aşımını yönetmek için)
CHROMA_PATH = "./chroma_db" # <-- Vektör DB'nin kaydedileceği klasör
LM_STUDIO_API_URL = "http://localhost:1234/v1/embeddings"
LOCAL_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5" # Başarıyla test ettiğiniz model adı

@st.cache_resource
def load_and_embed_knowledge_base():
    """
    1. ChromaDB'yi kontrol eder. Veri varsa, API çağrısı yapmadan yükler.
    2. Veri yoksa CSV'leri okur, API ile embedding oluşturur ve ChromaDB'ye kaydeder.
    """
    
    # 1. ChromaDB İstemcisini Başlat ve Koleksiyonu Al/Oluştur
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = "coffee_rag_data"
    collection = client.get_or_create_collection(name=collection_name)
    
    # 2. **KOTA KORUMASI:** ChromaDB'de kayıtlı veri var mı kontrol et
    if collection.count() > 0:
        count = collection.count()
        print(f"ChromaDB'den {count} adet belge yüklendi (API çağrısı atlandı).")
        st.info(f"Vektör veritabanından {count} belge yüklendi. (API Kotası Korundu)")
        
        # Tüm verileri (metin ve vektör) koleksiyondan çek
        all_data = collection.get(
            ids=collection.get()['ids'], 
            include=['documents', 'embeddings']
        )
        
        # RAG pipeline'ının beklediği in-memory formatına geri dönüştür
        KNOWLEDGE_BASE = all_data['documents']
        knowledge_base_embeddings = np.array(all_data['embeddings']) 
        
        # API çağrısı yapmadan ve yeniden hesaplama yapmadan fonksiyondan çık
        return KNOWLEDGE_BASE, knowledge_base_embeddings 
        
    # --- KOLEKSİYON BOŞ İSE: CSV OKU VE EMBEDDING OLUŞTUR (YENİDEN HESAPLAMA) ---
    
    KNOWLEDGE_BASE = []
    data_files_path = "data/*.csv" 
    
    print("ChromaDB boş. CSV Dosyaları okunuyor ve RAG belgelerine dönüştürülüyor...")
    # ... (CSV okuma ve KNOWLEDGE_BASE doldurma mantığı aynı kalır) ...
    for file_path in glob.glob(data_files_path):
        try:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8', keep_default_na=False)
            for index, row in df.iterrows():
                content = create_rag_document_from_row(row)
                if content.strip():
                    KNOWLEDGE_BASE.append(content)
        except Exception as e:
            print(f"HATA: {file_path} okunurken veya işlenirken sorun oluştu: {e}")

    if not KNOWLEDGE_BASE:
        st.error("HATA: Hiç dolu CSV satırı bulunamadı. 'data' klasöründeki CSV'yi kontrol edin.")
        return None, None

    print(f"Başarıyla {len(KNOWLEDGE_BASE)} adet RAG belgesi yüklendi.")
    st.info(f"{len(KNOWLEDGE_BASE)} belge için Yerel LM Studio API ile embedding oluşturuluyor...")

    all_embeddings = []

    # KNOWLEDGE_BASE'i küçük parçalara böl (Batching ve Rate Limiting)
    for i in range(0, len(KNOWLEDGE_BASE), BATCH_SIZE):
        batch = KNOWLEDGE_BASE[i:i + BATCH_SIZE]
        print(f"Toplu işlem {i//BATCH_SIZE + 1} başlatılıyor ({len(batch)} belge)...")
        
        try:
            # 1. YEREL LM STUDIO İÇİN GÖVDEYİ OLUŞTUR
            payload = {
                "model": LOCAL_EMBEDDING_MODEL, 
                "input": batch # LM Studio API'si genellikle liste formatını destekler
            }
            headers = {"Content-Type": "application/json"}

            # 2. requests.post İLE YEREL API İSTEĞİNİ GÖNDER
            response = requests.post(
                LM_STUDIO_API_URL, 
                headers=headers, 
                data=json.dumps(payload)
            )
            response.raise_for_status() # HTTP hatalarını yakala
            
            data = response.json()

            # 3. EMBEDDING'LERİ ÇIKAR VE TOPLA
            if data and 'data' in data:
                # Gelen yanıtta birden fazla embedding bulunur (batch kadar)
                for item in data['data']:
                    all_embeddings.append(item['embedding'])
            else:
                raise Exception("Yerel API'den geçerli embedding verisi alınamadı.")
                
            print(f"Toplu işlem {i//BATCH_SIZE + 1} tamamlandı. {SLEEP_TIME} saniye bekleniyor...")
            
            # 4. Yerel sunucuda kararlılık için bekleme
            time.sleep(SLEEP_TIME) 
            
        except Exception as e:
            # Hata oluşursa Streamlit'te göster ve None döndür
            st.error(f"Embedding oluşturulurken kritik hata (Toplu İşlem {i//BATCH_SIZE + 1}): {e}")
            print(f"Hata detayı: {e}")
            return None, None
# 3. BAŞARILI HESAPLAMA SONRASI ChromaDB'ye Kaydet
    print("Tüm embedding'ler oluşturuldu. ChromaDB'ye kalıcı olarak kaydediliyor...")
    doc_ids = [f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))]
    
    try:
        collection.add(
            embeddings=all_embeddings,
            documents=KNOWLEDGE_BASE,
            ids=doc_ids
        )
        print("Veri ChromaDB'ye başarıyla kaydedildi.")
        st.success("Embedding'ler oluşturuldu ve bir sonraki çalıştırma için kaydedildi.")
        
    except Exception as e:
        st.error(f"ChromaDB'ye kaydederken hata oluştu: {e}")
        # Kayıt hatası olsa bile bellek içi veriyi döndür
        knowledge_base_embeddings = np.array(all_embeddings)
        return KNOWLEDGE_BASE, knowledge_base_embeddings

    # <-- İŞLEM TAMAMLANDIĞINDA EKLENMESİ GEREKEN BAŞARILI DÖNÜŞ BURADA!
    # Yeni oluşturulan veriyi ve embedding'i döndür
    knowledge_base_embeddings = np.array(all_embeddings)
    print(f"DEBUG: YÜKLENEN BİLGİ TABANI Vektör Boyutu: {knowledge_base_embeddings.shape[1]}")
    return KNOWLEDGE_BASE, knowledge_base_embeddings # <-- Bu satır EKSİKTİ!

# Kaynakları yükle (Bu fonksiyon sadece ilk çalıştırmada çalışır)
KNOWLEDGE_BASE, knowledge_base_embeddings = load_and_embed_knowledge_base()

# --- 2. RAG FONKSİYONLARI (DEĞİŞİKLİK YOK) ---
# ... (retrieve_context, generate_response, simple_rag_pipeline fonksiyonları aynı kalır)



# Global Chroma Collection objesi (load_and_embed_knowledge_base'den yüklenecek)
CHROMA_COLLECTION = None
COLLECTION_NAME = "coffee_rag_data"
HEADERS = {"Content-Type": "application/json"}


def retrieve_context(query, top_k=13):
    """
    LM Studio'dan sorgu embedding'ini alır ve ChromaDB'ye sorgu gönderir.
    """
    global CHROMA_COLLECTION
    
    # Global koleksiyonu kontrol et
    if CHROMA_COLLECTION is None:
        # ChromaDB koleksiyonunu bir kez yükle (Eğer load_and_embed_knowledge_base güncellendiyse)
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            CHROMA_COLLECTION = client.get_collection(name=COLLECTION_NAME)
        except Exception:
             return "Hata: ChromaDB koleksiyonuna erişilemiyor. Lütfen uygulamayı yeniden başlatın."


    try:
        # 1. SORGUNUN EMBEDDING'İNİ OLUŞTUR (LM Studio ile)
        payload = {
            "model": LOCAL_EMBEDDING_MODEL,
            "input": query
        }
        
        response = requests.post(
            LM_STUDIO_API_URL, 
            headers=HEADERS, 
            data=json.dumps(payload)
        )
        response.raise_for_status() 
        
        data = response.json()
        
        if not (data and 'data' in data and data['data']):
            raise Exception("Yerel API'den geçerli sorgu embedding'i alınamadı.")

        # Tek bir sorgu vektörünü al
        query_vector = data['data'][0]['embedding']

        # 2. CHROMADB'DE EN YAKIN BELGELERİ ARA
        results = CHROMA_COLLECTION.query(
            query_embeddings=[query_vector], # Chroma'ya vektörü gönder
            n_results=top_k, 
            include=['documents', 'distances'] # Metni ve mesafeyi döndür
        )

        # 3. Bağlamı formatla
        retrieved_chunks = results['documents'][0]
        
        print(f"\n[Geri Alınan Bağlam ({top_k} adet)]:")
        for chunk in retrieved_chunks:
            print(f"- {chunk[:200]}...") 
            
        return "\n".join(retrieved_chunks)
        
    except Exception as e:
        print(f"Hata (retrieve_context): {e}")
        return f"Bağlam alınırken bir hata oluştu: {e}"
from google.genai.types import GenerationConfig
def generate_response(query, context, history=None, temperature=0.5):
    """Geri alınan bağlamı ve sohbet geçmişini kullanarak Gemini'den yanıt üretir (Üretim)."""
    
    # Sohbet geçmişini (history) formatlama
    history_str = ""
    if history:
        history_str += "\n\n--- SOHBET GEÇMİŞİ ---\n"
        for message in history:
            role = "Kullanıcı" if message["role"] == "user" else "Kahve Sever"
            history_str += f"{role}: {message['content']}\n"
        history_str += "-------------------------\n\n"
        
    try:
        # Prompt'a geçmişi dahil edin
        prompt = f"""
        Sen 'Kahve Sever' adında bir uzmansın.
        Görev: Aşağıdaki 'SOHBET GEÇMİŞİ' ve 'BAĞLAM' bölümlerinde sağlanan bilgilere dayanarak '{query}' sorusuna en uygun yanıtı verin.
        
        {history_str}
        
        Aşağıdaki 'BAĞLAM' bölümünde sağlanan bilgilere dayanarak '{query}' sorusuna yanıt verin.
        Eğer BAĞLAM'da veya SOHBET GEÇMİŞİ'nde yeterli bilgi yoksa, 'Üzgünüm, bu kahve bilgisine sahip değilim.' diye cevap verin.

        BAĞLAM:
        {context} 

        SORU: {query}
        YANIT:
        """
        
        print(f"\n[Yanıt Üretiliyor - Temperature: {temperature}]")
        generation_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # *** DEĞİŞİKLİK 1: Configuration nesnesi oluşturuldu ***
        generation_config_dict = {
            "temperature": temperature
            # İleride max_output_tokens, top_k gibi ayarlar eklemek isterseniz buraya ekleyin.
        }
        
        response = generation_model.generate_content(
            prompt, 
            generation_config=generation_config_dict # Sözlüğü doğrudan ilet
        )

        return response.text
    except Exception as e:
        print(f"Hata (generate_response): {e}")
        return f"Yanıt üretilirken bir hata oluştu: {e}"


def simple_rag_pipeline(query, history=None, temperature=0.5):
    """Tüm RAG sürecini çalıştırır ve SADECE nihai yanıtı döndürür."""
    # NOT: retrieve_context fonksiyonunun da temperature parametresini alması gerekir
    # ancak RAG araması için bu parametre gereksiz olduğu için şimdilik atlıyoruz.
    # Eğer retrieve_context'in imzasına eklediyseniz, burada da alıp iletmelisiniz.
    context = retrieve_context(query) 
    
    # Sohbet geçmişini (history) ve YENİ PARAMETRE temperature'ı generate_response'a iletin
    final_response = generate_response(
        query, 
        context, 
        history, 
        temperature=temperature # Streamlit'ten gelen değeri ilet
    ) 
    return final_response

# --- SICAKLIK KADEMELERİ TANIMI (Önceki cevaptan) ---
TEMPERATURE_KADEMELERI = {
    0.00: "🔬 Ciddi & Deterministik (Robot gibi kesin cevaplar, yaratıcılık yok.)",
    0.25: "📝 Düşük Yaratıcılık (Genellikle kesin, nadiren farklı kelimeler kullanır.)",
    0.50: "🧠 Dengeli & Standart (İyi bir denge, çoğu soru için önerilir.)",
    0.75: "🎨 Yüksek Yaratıcılık (Farklı kelimeler, benzersiz ifadeler kullanır. Halüsinasyon riski artar.)",
}

# Slider'ın alabileceği değerler (sözlüğün anahtarları)
ALLOWED_VALUES = list(TEMPERATURE_KADEMELERI.keys())
DEFAULT_VALUE = 0.50 


# --- 1. SİDEBAR VE SICAKLIK AYARI ---

st.set_page_config(page_title="Coffeephil", page_icon="☕")
st.title("☕ Coffeephil")
st.caption("Nitelikli kahve demleme yöntemleri hakkında her şeyi bana sorabilirsiniz.")

# Kenar Çubuğuna Sıcaklık Ayarı Ekleme
with st.sidebar:
    st.header("Model Ayarları")
    
    # Sıcaklık Slider'ı
    selected_temperature = st.slider(
        label="Sıcaklık (Temperature)",
        min_value=min(ALLOWED_VALUES),
        max_value=max(ALLOWED_VALUES),
        value=DEFAULT_VALUE,
        step=0.25, # Sadece kademeli değerleri seçmeyi garantiler.
        key="llm_temperature_slider", # Session state'de saklanması için bir anahtar
        help="Modelin cevap üretirken ne kadar yaratıcı ve rastgele olacağını belirler."
    )
    
    # Seçilen sıcaklığın açıklamasını dinamik olarak gösteren kısım
    st.markdown("---")
    st.subheader("Seçilen Cevap Tarzı")
    description = TEMPERATURE_KADEMELERI.get(selected_temperature, "Bilinmeyen Ayar")
    
    st.markdown(
        f"**Sıcaklık Değeri:** `{selected_temperature}`\n\n"
        f"**Tarz:** {description}"
    )


# --- 2. CHAT ARAYÜZÜ VE RAG MANTIĞI ---

# Sohbet geçmişini session_state'de sakla
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni bir girdi al
if prompt := st.chat_input("Nitelikli kahve demleme teknikleri ile ilgili bir soru sorun..."):
    
    # 1. Kullanıcının mesajını sohbet geçmişine ekle ve ekranda göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Yeni eklendi: Son kullanıcı sorgusu HARİÇ tüm geçmişi al.
    history_for_rag = st.session_state.messages[:-1] 
    
    # 2. Chatbot'un yanıtını al
    with st.spinner("Kahve çekirdeklerini inceliyorum..."):
        # ÖNEMLİ DEĞİŞİKLİK: Sıcaklık ayarını simple_rag_pipeline'a iletiyoruz.
        response = simple_rag_pipeline(
            query=prompt, 
            history=history_for_rag,
            temperature=selected_temperature # Streamlit'ten alınan değeri ilet
        )
    
    # 3. Chatbot'un yanıtını ekranda göster
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # 4. Chatbot'un yanıtını sohbet geçmişine ekle
    st.session_state.messages.append({"role": "assistant", "content": response})