Customer Support LLM Fine-Tuning 

Bu projede, müşteri destek (customer support) alanına yönelik bir büyük dil modeli (LLM) özelleştirilmiştir. Proje kapsamında geliştirilen uygulamanın ana hedefi, kullanıcıdan gelen müşteri mesajını alıp uygun destek yanıtı üretmektir.

Projede kullanılacak veri seti belirli senaryoları içerecek şekilde sentetik olarak JSONL formatında oluşturuldu. Her bir satır eğitim örneği olacak şekilde tasarlanmıştır. Veri üretiminde amaç, modelin sadece kibar bir cevap üretmesi değil, aynı zamanda problem türünü ayırt edebilmesi ve senaryoya uygun aksiyon adımları önerebilmesidir. Bu nedenle veri tek tik kalıplar yerine farklı uzunluklarda ve farklı stratejilerde cevaplar içerecek şekilde hazırlanmıştır. 

Veri setinde müşteri destek alanında yaygın görülen senaryolar:
-Teslimat gecikmesi (delivery delay)

-Hasarlı ürün (damaged item)

-Yanlış ürün gönderimi (wrong item)

-Çifte ücretlendirme (double charge)

-İade / geri ödeme durumu (refund status)

-Abonelik iptali (cancel subscription)

-Hesaba giriş sorunu (login issue)

-Şifre sıfırlama sorunu (password reset)

-İnternet bağlantısında kopma (internet drops)

-Uygulama çökmesi (app crash)


Her senaryo için birden fazla müşteri cümle şablonu ve birden fazla destek yanıt şablonu hazırlanmıştır. Böylece model aynı problemi farklı ifade ediliş biçimleriyle görmüş ve genelleme yeteneğini arttıracak örnek çeşitliliğini elde etmiştir. Cevap tarafında da sadece "özür dileriz" gibi genel ifadeler değil, gerekli ise müşteri bilgisini isteme gerekli ise adım adım yönlendirme yapma, gerekli ise iade/geri ödeme süresi gibi süreç bilgisini verme gibi müşteri destek akışına uygun içerikler üretilmiştir. 

MODEL SEÇİMİ

Bu projede base model olarak instruct uyumlu, küçük/orta ölçekli bir LLM olan Qwen2.5-3B Instruct tercih edilmiştir. Bu model seçimi hem müşteri destek türünde diyalog üretimine uygun olması hem de Colab GPU üzerinde LoRA ile pratik sürelerde fine-tune edebilmesi açısından uygundur.

Fine-tune yöntemi olarak "Unsloth-Ai" resmi internet sayfası üzerinden elde edilen notebook ile SFT (Supervised Fine-Tuning) tercih edilmiştir. Unsloth-Ai sayesinde Google Colab üzerinden arayüz yardımıyla fine-tune işlemleri gerçekleştirildi. SFT yaklaşımında model, her eğitim örneğinde verilen müşteri mesajına karşılık veri setinde bulunan "doğru" destek yanıtını öğrenir. Böylece model eğitim verisindeki format ve alan bilgisini kendi üretimlerine yansıtabilir.

Bu projede tam model güncellemesi yerine LoRA (Low-Rank Adaptation) kullanılmıştır. LoRA modelin tüm ağırlıklarını güncelemek yerine belirli katmanlara küçük düşük-rank adaptasyon ağırlıkları ekleyerek eğitim yapar. Bu yöntem eğitimi hızlandırmak ve kaynak tüketimini azalatmak için etkilidir. 

EĞİTİM

-3 epoch Sentetik verinin senaryolarını modelin yeterince görmesi
-Learning rate: 1e-4 gibi daha stabil bi değer
-Gradient accumulation: efektif batch'i artırmak
-BF16: A100 üzerinde stabil ve hızlı eğitim

Eğitim sırasında log'lar takip edilmiş ve modelin loss değerinin zamanla azalması ve eğitim akışının stabil ilerlemesi başarı kriteri olarak izlenmiştir.

Eğitim sonunda kaydedilen çıktı LoRA adapter ağırlıkları ve tokenizer dosyalarıdır. Kaydetme işlemi Drive'e yapılmıştır. Bu yaklaşımın ile modelin başka ortamlarda tekrar tekrar kullanılabilir olmasıdır. 

Modelin kullanıcı tarafında denenebilir olması için Gradio Arayüzü tercih edildi. Arayüz üzerinde yapılan testler ile modelin performansı değerlendirildi. Yapılan testler gösterdi ki eğitilen model gelen mesajdaki senaryoyu doğru ayırmakta ve uygun metni oluşturmakta. Örneğin çifte ücretlendirme sorularında "pending authorization/settlement/refund" gibi süreç bazlı ifadeler, teslimat gecikmesinde "tracking/investigation/carrier update" gibi terimler üretmesi modelin alan bilgisini doğru öğrendiğini göstermiştir.
Bununla birlikte çıktıların bir kısmında tekrar eden nezaket cümleleri ("happy to help", "thank you for your patience" vb) görülmüştür. Bu, denetimli fine-tune' da çok sık rastlanan bir durumdur. Model veri setinde sık geçen kalıpları genelleyerek üretir. Bu durum daha fazla veri çeşitliliği veya inference sırasında daha güçlü tekrar engelleyici ayarlarla azaltılabilir. Proje kapsamında amaçlanan müşteri destek asistanı prototipi elde edilmiştir. Yapılan testlere ait görsellerden bazıları aşağıda verilmiştir. 



### Billing Issue Example
![Double charge example](img/double_charge.png)

### Shipping Delay Example
![Delivery delay example](img/delivery_delay.png)

### Account Login Issue Example
![Login issue example](img/login_issue.png)


### Modeli arayüz üzerinde kullanmak için main.ipynb dosyasındaki kodlar sırasıyla çalıştırılmalıdır.
