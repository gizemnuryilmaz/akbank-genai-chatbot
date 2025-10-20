import streamlit as st
import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. YÜKLEME VE KURULUM ---

# .env dosyasındaki API anahtarını yükle
load_dotenv() 
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
try:
    # API anahtarını ortam değişkeninden alarak ayarla
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except KeyError:
    st.error("Hata: GEMINI_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")
    st.stop()
except Exception as e:
    st.error(f"API anahtarı ayarlanırken bir hata oluştu: {e}")
    st.stop()

# Embedding modeli
EMBEDDING_MODEL = 'gemini-embedding-001' # Gemini için önerilen embedding modeli

# Streamlit'in @st.cache_resource dekoratörü, bu fonksiyonun sadece bir kez
# çalışmasını sağlar. Veri yükleme ve embedding oluşturma gibi ağır
# işlemleri her seferinde tekrar yapmamak için bu şart!
@st.cache_resource
def load_and_embed_knowledge_base():
    """
    Veri dosyalarını okur, boş olanları filtreler ve
    tüm veritabanı için embedding'leri oluşturur.
    """
    KNOWLEDGE_BASE = []
    data_files_path = "data/*.txt"
    
    print("Dosyalar okunuyor ve filtreleniyor...")
    for file_path in glob.glob(data_files_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    KNOWLEDGE_BASE.append(content)
                else:
                    print(f"UYARI: {file_path} dosyası boş ve atlandı.")
        except Exception as e:
            print(f"HATA: {file_path} okunurken sorun oluştu: {e}")

    if not KNOWLEDGE_BASE:
        st.error("HATA: Hiç dolu .txt dosyası bulunamadı. 'data' klasörünü kontrol edin.")
        return None, None

    print(f"Başarıyla {len(KNOWLEDGE_BASE)} adet dolu doküman yüklendi.")
    print("Veri setinin embedding'leri oluşturuluyor...")
    
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=KNOWLEDGE_BASE,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = np.array(response['embedding'])
        print(f"Oluşturulan embedding sayısı: {len(embeddings)}")
        return KNOWLEDGE_BASE, embeddings
    except Exception as e:
        st.error(f"Embedding oluşturulurken kritik hata: {e}")
        return None, None

# Kaynakları yükle (Bu fonksiyon sadece ilk çalıştırmada çalışır)
KNOWLEDGE_BASE, knowledge_base_embeddings = load_and_embed_knowledge_base()

# --- 2. RAG FONKSİYONLARI ---

def retrieve_context(query, top_k=2):
    """Sorgu ile en alakalı veri parçalarını bulur (Geri Alma)."""
    if knowledge_base_embeddings is None:
        return "Hata: Bilgi tabanı (embeddings) boş."
        
    try:
        query_embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=[query],
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array(query_embedding_response['embedding'])
        
        similarities = cosine_similarity(query_embedding, knowledge_base_embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        retrieved_chunks = [KNOWLEDGE_BASE[i] for i in top_indices]
        print(f"\n[Geri Alınan Bağlam ({top_k} adet)]:")
        for chunk in retrieved_chunks:
            print(f"- {chunk[:200]}...") 
            
        return "\n".join(retrieved_chunks)
    except Exception as e:
        print(f"Hata (retrieve_context): {e}")
        return f"Bağlam alınırken bir hata oluştu: {e}"

def generate_response(query, context):
    """Geri alınan bağlamı kullanarak Gemini'den yanıt üretir (Üretim)."""
    try:
        prompt = f"""
        Sen 'Kahve Gurusu' adında bir uzmansın.
        Aşağıdaki 'BAĞLAM' bölümünde sağlanan bilgilere dayanarak '{query}' sorusuna yanıt verin.
        Eğer bağlamda yeterli bilgi yoksa, 'Üzgünüm, bu kahve bilgisine sahip değilim.' diye cevap verin.

        BAĞLAM:
        {context} 

        SORU: {query}
        YANIT:
        """
        
        print("\n[Yanıt Üretiliyor...]")
        generation_model = genai.GenerativeModel('gemini-2.0-flash')
        response = generation_model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"Hata (generate_response): {e}")
        return f"Yanıt üretilirken bir hata oluştu: {e}"

def simple_rag_pipeline(query):
    """Tüm RAG sürecini çalıştırır ve SADECE nihai yanıtı döndürür."""
    context = retrieve_context(query)
    final_response = generate_response(query, context)
    return final_response

# --- 3. STREAMLIT WEB ARAYÜZÜ ---

st.set_page_config(page_title="Coffeephil", page_icon="☕")
st.title("☕ Coffeephil")
st.caption("Nitelikli kahve demleme yöntemleri hakkında her şeyi bana sorabilirsiniz.")

# Sohbet geçmişini session_state'de sakla
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni bir girdi al
if prompt := st.chat_input("Nitelikli kahve demleme teknikleri ile ilgili bir soru sorun..."):
    
    # Kullanıcının mesajını sohbet geçmişine ekle ve ekranda göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Chatbot'un yanıtını al
    with st.spinner("Kahve çekirdeklerini inceliyorum..."):
        response = simple_rag_pipeline(prompt)
    
    # Chatbot'un yanıtını ekranda göster
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Chatbot'un yanıtını sohbet geçmişine ekle
    st.session_state.messages.append({"role": "assistant", "content": response})