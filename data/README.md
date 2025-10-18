# Data Klasörü

Bu klasör, projede kullanılan metin tabanlı kaynakları içerir. Dosyalar kahve hazırlama yöntemleri, saklama ve öğütme ile ilgili rehberler ve kahve tarihçesi gibi kısa metinler içerir. Bu içerikler, bir sohbet botunun bilgi tabanını oluşturmak veya metin işleme (NLP) deneyleri için kullanılabilir.

Tüm dosyalar UTF-8 kodlamasıyla saklanmıştır.

## Dosyalar ve Açıklamaları

- `aeropress_rehberi.txt` — AeroPress ile kahve demleme adımları ve ipuçları.
- `chemex_rehberi.txt` — Chemex yöntemiyle filtre kahve demleme rehberi.
- `espresso_rehberi.txt` — Espresso hazırlama temelleri ve ekipman notları.
- `filtre_kahve_rehberi.txt` — Genel filtre kahve (drip) demleme teknikleri.
- `french_press_rehberi.txt` — French Press (plunger) ile demleme yöntemi.
- `hario_v60_rehberi.txt` — Hario V60 yöntemi ve ideal parametreler.
- `kahve_çekirdeğinin_tarihçesi.txt` — Kahve çekirdeğinin kısa tarihçesi ve coğrafi bilgiler.
- `kahveyi_saklamanın_ideal_yöntemleri.txt` — Kahve çekirdeği ve öğütülmüş kahvenin saklama tavsiyeleri.
- `mokapot_rehberi.txt` — Moka Pot (İtalyan kahve makinesi) kullanımı.
- `öğütme_ve_su.txt` — Öğütme dereceleri, su sıcaklığı ve oranlar hakkında bilgiler.
- `türk_kahvesi_rehberi.txt` — Türk kahvesi hazırlama adımları ve kültürel notlar.
- `v60_rehberi.txt` — V60 (benzer Hario V60) için kısa bir özet.

> Not: Bazı dosyalar isim benzerliği nedeniyle `hario_v60_rehberi.txt` ve `v60_rehberi.txt` ayrı dosyalar olarak bulunuyor; içeriğin tekrarını azaltmak isterseniz birleştirilebilir.

## Kullanım Örnekleri

Python ile bu klasördeki tüm metin dosyalarını okumak ve basit bir işlem (örneğin token sayımı) yapmak için örnek bir kod:

```python
from pathlib import Path

data_dir = Path(__file__).parent

for txt in sorted(data_dir.glob('*.txt')):
    text = txt.read_text(encoding='utf-8')
    words = text.split()
    print(f"{txt.name}: {len(words)} kelime, {len(text)} karakter")
```

Basit bir sohbet botu veya bilgi alma (retrieval) pipeline'ında bu metinler doğrudan kaynak dokümanlar olarak kullanılabilir. Önerilen adımlar:

1. Dosyaları parçalara (chunk) bölün ve her parçaya bir kimlik verin.
2. Her parçayı vektörlere dönüştürün (embedding).
3. Kullanıcı sorgusu ile en alakalı parçaları getirip bir model ile cevap oluşturun.

## Kodlama ve Temizlik

- Dosyalar UTF-8 formatındadır. Yeni dosya eklerken aynı kodlamayı kullanın.
- Dosya adlarında Türkçe karakter kullanmaktan kaçınmak, platformlar arası uyumluluk için önerilir. Mevcut dosyalarda Türkçe karakterler bulunmakta; isterseniz README veya bir script ile isimlendirme standardize edilebilir.

## Katkıda Bulunma

Yeni bir rehber eklemek veya mevcut bir dosyayı güncellemek isterseniz:

1. Yeni bir `.txt` dosyası oluşturun ve açıklayıcı bir ad verin (örn. `soğuk_demleme_rehberi.txt`).
2. İçeriği açık ve kısa paragraflar halinde yazın.
3. Değişiklik yaparken bir PR açın; PR açıklamasında kaynaklar/alıntılar varsa belirtin.

## Lisans ve Atıf

Bu klasördeki metinler proje ile birlikte dağıtılır. Eğer dış kaynaklardan alıntı yapıldıysa, orijinal kaynağa atıf yapılmalıdır. Projenin genel lisansı `README.md` dosyasında belirtilmiştir; metin içerikleri için ek kısıtlamalar varsa PR içinde belirtin.

---

Hazırlayan: proje dosyaları otomatik açıklama scripti (2025)
