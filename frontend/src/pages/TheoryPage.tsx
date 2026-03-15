import React, { useState } from 'react';

const TheoryPage = () => {
  const [activeChapter, setActiveChapter] = useState('nlp');

  const chapters = [
    { id: 'nlp', title: '1. Tiền xử lý (underthesea)', icon: '🧼' },
    { id: 'vector', title: '2. Vectơ hóa (TF-IDF & Doc2Vec)', icon: '🔢' },
    { id: 'models', title: '3. 5 Thuật toán Phân loại AI', icon: '🧠' },
    { id: 'metrics', title: '4. Chỉ số Đánh giá (Metrics)', icon: '📈' },
    { id: 'matrix', title: '5. Ma trận Nhầm lẫn (Confusion)', icon: '🧱' },
  ];

  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-[#fcfcfd] text-slate-800 font-sans">
      {/* Sidebar - Cố định bên trái */}
      <div className="w-full md:w-80 bg-white border-r border-slate-200 p-8 space-y-3 sticky top-0 h-fit md:h-screen overflow-y-auto shadow-sm">
        <div className="mb-10">
          <h2 className="text-2xl font-black text-slate-900 tracking-tighter uppercase leading-tight">
            AI Knowledge <br/><span className="text-blue-600">Base</span>
          </h2>
          <div className="h-1 w-12 bg-blue-600 mt-2"></div>
        </div>
        {chapters.map((ch) => (
          <button
            key={ch.id}
            onClick={() => setActiveChapter(ch.id)}
            className={`w-full text-left px-5 py-4 rounded-2xl text-sm font-black transition-all flex items-center gap-4 border-2 ${
              activeChapter === ch.id 
                ? 'bg-blue-600 text-white border-blue-600 shadow-xl shadow-blue-100 scale-105' 
                : 'text-slate-400 border-transparent hover:bg-slate-50 hover:text-slate-600'
            }`}
          >
            <span className="text-xl">{ch.icon}</span> {ch.title}
          </button>
        ))}
      </div>

      {/* Nội dung chính */}
      <div className="flex-1 p-6 md:p-16 max-w-5xl mx-auto overflow-y-auto">
        
        {/* CHAPTER 1: UNDERTHESEA */}
        {activeChapter === 'nlp' && (
          <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h1 className="text-5xl font-black mb-6 text-slate-900 tracking-tight">Tiền xử lý văn bản với <span className="text-blue-600">underthesea</span></h1>
              <p className="text-xl text-slate-500 leading-relaxed">
                Máy tính không hiểu ngôn ngữ như con người. Trước khi đưa vào AI, văn bản cần được "phẫu thuật" để loại bỏ nhiễu và chuẩn hóa cấu trúc.
              </p>
            </section>

            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 space-y-8">
              <div>
                <h3 className="text-2xl font-black mb-4 flex items-center gap-3">
                   <div className="w-2 h-8 bg-blue-600 rounded-full"></div> 1. Tại sao phải tách từ (Tokenization)?
                </h3>
                <p className="text-slate-600 leading-relaxed mb-4">
                  Tiếng Việt có đặc trưng là <strong>từ ghép</strong>. Nếu cắt theo khoảng trắng, từ "thành phố" sẽ bị tách thành "thành" và "phố", làm mất hoàn toàn ý nghĩa gốc.
                </p>
                <div className="bg-slate-900 p-8 rounded-3xl text-white font-mono text-sm shadow-2xl relative">
                  <p className="text-slate-500 mb-4">// Logic của underthesea</p>
                  <p className="mb-2"><span className="text-pink-400">Input:</span> "Học sinh học sinh học rất giỏi"</p>
                  <p><span className="text-green-400">Output:</span> ["Học_sinh", "học", "sinh_học", "rất", "giỏi"]</p>
                  <div className="absolute bottom-4 right-8 text-[10px] text-slate-600 font-black uppercase">Vietnamese Word Segmentation</div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-black mb-4 flex items-center gap-3">
                   <div className="w-2 h-8 bg-blue-600 rounded-full"></div> 2. Quy trình làm sạch (Data Cleaning)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { t: 'Lowercase', d: 'Chuyển "TIN GIẢ" thành "tin giả". Giúp mô hình không bị phân tâm bởi chữ hoa/thường.' },
                    { t: 'Regex Cleaning', d: 'Xóa ký tự đặc biệt (!, @, #, $, ...) và các đường link (http://...).' },
                    { t: 'Stopwords', d: 'Loại bỏ các từ vô nghĩa như "thì, là, mà, của..." để tập trung vào từ mang nội dung.' }
                  ].map((item, i) => (
                    <div key={i} className="p-6 bg-slate-50 rounded-3xl border border-slate-100">
                      <h4 className="font-black text-blue-600 text-xs uppercase mb-2 tracking-widest">{item.t}</h4>
                      <p className="text-sm text-slate-600 font-medium leading-relaxed">{item.d}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* CHAPTER 2: VECTORIZATION */}
        {activeChapter === 'vector' && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h1 className="text-5xl font-black mb-6 text-slate-900 tracking-tight">Vectơ hóa: Biến chữ thành <span className="text-blue-600">Toán học</span></h1>
              <p className="text-xl text-slate-500 leading-relaxed italic">AI không đọc chữ, nó tính toán trên các không gian vectơ hàng trăm chiều.</p>
            </section>

            {/* TF-IDF */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 space-y-8">
              <div className="flex justify-between items-start">
                <h2 className="text-3xl font-black">1. TF-IDF (Term Frequency - Inverse Document Frequency)</h2>
                <span className="px-4 py-1 bg-blue-100 text-blue-700 rounded-full text-[10px] font-black uppercase">Thống kê học</span>
              </div>
              <p className="text-slate-600 leading-relaxed">
                Đây là kỹ thuật đánh giá tầm quan trọng của một từ dựa trên tần suất xuất hiện. Một từ được coi là quan trọng nếu nó xuất hiện nhiều trong 1 văn bản cụ thể nhưng lại hiếm gặp trong toàn bộ kho dữ liệu.
              </p>
              
              <div className="bg-blue-50 p-8 rounded-3xl border border-blue-100">
                <h4 className="text-xs font-black text-blue-400 uppercase mb-6 tracking-widest text-center">Công thức toán học chi tiết</h4>
                <div className="space-y-8">
                  <div className="flex flex-col md:flex-row items-center justify-around gap-8">
                    <div className="text-center">
                      <div className="text-2xl font-serif mb-2 bg-white px-6 py-3 rounded-xl shadow-sm italic">TF(t, d) = n<sub>t,d</sub> / Σn<sub>k,d</sub></div>
                      <p className="text-[10px] font-bold text-slate-400">Tần suất từ t trong văn bản d</p>
                    </div>
                    <div className="text-2xl font-bold text-blue-300">×</div>
                    <div className="text-center">
                      <div className="text-2xl font-serif mb-2 bg-white px-6 py-3 rounded-xl shadow-sm italic">IDF(t) = log(N / df<sub>t</sub>)</div>
                      <p className="text-[10px] font-bold text-slate-400">Nghịch đảo tần suất văn bản</p>
                    </div>
                  </div>
                  <div className="h-px bg-blue-100 w-full"></div>
                  <p className="text-sm text-blue-900/70 font-medium text-center">
                    <strong>Kết quả:</strong> Một từ có TF-IDF cao là từ khóa đặc trưng giúp phân biệt tin thật và tin giả nhanh chóng.
                  </p>
                </div>
              </div>
            </div>

            {/* Doc2Vec */}
            <div className="bg-indigo-900 p-10 rounded-[2.5rem] shadow-2xl text-white space-y-8 relative overflow-hidden">
               <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-20 -mt-20 blur-3xl"></div>
               <div className="flex justify-between items-start">
                <h2 className="text-3xl font-black">2. Doc2Vec (Paragraph Embeddings)</h2>
                <span className="px-4 py-1 bg-indigo-500 text-indigo-100 rounded-full text-[10px] font-black uppercase">Deep Learning</span>
              </div>
              <p className="text-indigo-100/80 leading-relaxed">
                Khác với TF-IDF chỉ đếm chữ, Doc2Vec sử dụng mạng Nơ-ron để học <strong>ngữ nghĩa (Semantic)</strong>. Nó hiểu được ngữ cảnh: nếu từ A và B thường đứng cạnh nhau, chúng có liên quan về nghĩa.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white/10 p-6 rounded-3xl border border-white/10">
                  <h4 className="font-black text-yellow-400 text-xs mb-4 uppercase">Nguyên lý Hoạt động</h4>
                  <p className="text-sm leading-relaxed">
                    Nó gán cho mỗi văn bản một "ID Vector" duy nhất. Trong quá trình học, nó cố gắng dự đoán từ tiếp theo dựa trên các từ xung quanh + ID Vector này. Kết quả là ID Vector sẽ mang "linh hồn" của toàn bộ bài báo.
                  </p>
                </div>
                <div className="bg-white/10 p-6 rounded-3xl border border-white/10">
                  <h4 className="font-black text-indigo-300 text-xs mb-4 uppercase">Tại sao nó mạnh?</h4>
                  <p className="text-sm leading-relaxed">
                    Nó bắt được các sắc thái biểu cảm. Tin giả thường có phong cách viết kích động, cường điệu. Doc2Vec nhận diện được "phong cách" này dù tác giả dùng từ ngữ khác nhau.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* CHAPTER 3: ALGORITHMS */}
        {activeChapter === 'models' && (
          <div className="space-y-16 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h1 className="text-5xl font-black mb-6 text-slate-900 tracking-tight">5 Thuật toán <span className="text-blue-600">Phân loại</span></h1>
              <p className="text-xl text-slate-500">Mỗi thuật toán là một "góc nhìn" khác nhau về dữ liệu.</p>
            </section>

            {/* Logistic Regression */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 border-l-[16px] border-l-blue-600">
              <h2 className="text-3xl font-black mb-4">1. Logistic Regression (Hồi quy Logistic)</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="space-y-4">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Dù tên là "Hồi quy", đây là thuật toán <strong>Phân loại</strong>. Nó tìm một đường thẳng (hoặc siêu phẳng) tốt nhất để chia dữ liệu thành 2 miền.
                  </p>
                  <div className="p-4 bg-blue-50 rounded-2xl">
                    <h5 className="text-[10px] font-black text-blue-600 uppercase mb-2">Ưu điểm</h5>
                    <p className="text-xs font-bold text-slate-700">Chạy cực nhanh, dễ hiểu kết quả thông qua các trọng số của từ khóa.</p>
                  </div>
                </div>
                <div className="bg-slate-900 p-6 rounded-3xl text-white">
                  <h5 className="text-[10px] font-black text-slate-500 uppercase mb-4 tracking-widest text-center">Hàm kích hoạt Sigmoid</h5>
                  <div className="text-xl italic text-center text-yellow-400 font-serif my-4">σ(z) = 1 / (1 + e<sup>-z</sup>)</div>
                  <p className="text-[10px] text-slate-400 text-center italic">Nén mọi giá trị về khoảng [0, 1] để đại diện cho xác suất.</p>
                </div>
              </div>
            </div>

            {/* SVM */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 border-l-[16px] border-l-purple-600">
              <h2 className="text-3xl font-black mb-4">2. Support Vector Machine (SVM)</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="space-y-4">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    SVM không chỉ chia dữ liệu, nó tìm ranh giới có <strong>khoảng cách (Margin) lớn nhất</strong> đến các điểm gần nhất (Support Vectors).
                  </p>
                  <div className="p-4 bg-purple-50 rounded-2xl">
                    <h5 className="text-[10px] font-black text-purple-600 uppercase mb-2">Sức mạnh</h5>
                    <p className="text-xs font-bold text-slate-700">Cực kỳ hiệu quả trong không gian cao chiều (như TF-IDF với hàng nghìn từ).</p>
                  </div>
                </div>
                <div className="bg-slate-900 p-6 rounded-3xl text-white flex flex-col justify-center">
                  <h5 className="text-[10px] font-black text-slate-500 uppercase mb-4 tracking-widest text-center">Mục tiêu tối ưu</h5>
                  <div className="text-center space-y-2">
                    <div className="text-lg font-bold">Maximize Distance = 2 / ||w||</div>
                    <div className="text-[10px] text-slate-400">w là vector trọng số của siêu phẳng.</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Decision Tree */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 border-l-[16px] border-l-green-600">
              <h2 className="text-3xl font-black mb-4">3. Decision Tree (Cây quyết định)</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="space-y-4">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Hoạt động như một chuỗi câu hỏi Yes/No. Nó tự tìm ra những từ khóa quan trọng nhất để làm "nút" rẽ nhánh.
                  </p>
                  <div className="p-4 bg-green-50 rounded-2xl">
                    <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Cơ chế</h5>
                    <p className="text-xs font-bold text-slate-700">Sử dụng <strong>Entropy</strong> (độ hỗn loạn) hoặc <strong>Gini Impurity</strong> để đo lường độ tinh khiết sau mỗi lần chia.</p>
                  </div>
                </div>
                <div className="bg-slate-900 p-6 rounded-3xl text-white flex flex-col justify-center font-mono text-[10px]">
                  <p className="text-green-400">if "miễn_phí" in text:</p>
                  <p className="ml-4 text-blue-400">if "trúng_thưởng" in text: return FAKE</p>
                  <p className="ml-4 text-blue-400">else: return REAL</p>
                  <p className="text-green-400">else: return REAL</p>
                </div>
              </div>
            </div>

            {/* Naive Bayes */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 border-l-[16px] border-l-orange-600">
              <h2 className="text-3xl font-black mb-4">4. Naive Bayes (Bayes ngây thơ)</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="space-y-4">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Dựa trên lý thuyết xác suất. Nó tính xác suất của một bài báo là Tin giả dựa trên xác suất của từng từ xuất hiện trong bài đó.
                  </p>
                  <div className="p-4 bg-orange-50 rounded-2xl">
                    <h5 className="text-[10px] font-black text-orange-600 uppercase mb-2">Tại sao "ngây thơ"?</h5>
                    <p className="text-xs font-bold text-slate-700">Vì nó giả định các từ xuất hiện độc lập với nhau (dù thực tế chúng có liên kết ngữ pháp).</p>
                  </div>
                </div>
                <div className="bg-slate-900 p-6 rounded-3xl text-white">
                  <h5 className="text-[10px] font-black text-slate-500 uppercase mb-4 tracking-widest text-center">Định lý Bayes</h5>
                  <div className="text-lg text-center text-orange-400 font-serif my-4">P(A|B) = [P(B|A) * P(A)] / P(B)</div>
                </div>
              </div>
            </div>

            {/* Random Forest */}
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100 border-l-[16px] border-l-red-600">
              <h2 className="text-3xl font-black mb-4">5. Random Forest (Rừng ngẫu nhiên)</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="space-y-4">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Là phương pháp **Ensemble Learning** (Học kết hợp). Thay vì tin vào 1 cây quyết định, nó xây dựng hàng trăm cây và lấy kết quả theo số đông.
                  </p>
                  <div className="p-4 bg-red-50 rounded-2xl">
                    <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Triết lý</h5>
                    <p className="text-xs font-bold text-slate-700">"Sự thông thái của đám đông". Nó rất bền bỉ và khó bị đánh lừa bởi dữ liệu nhiễu.</p>
                  </div>
                </div>
                <div className="bg-slate-900 p-6 rounded-3xl text-white flex items-center justify-center gap-4">
                   <div className="text-center">
                     <div className="text-2xl">🌳🌳🌳</div>
                     <div className="text-[10px] mt-2 font-black uppercase">Voting System</div>
                   </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* CHAPTER 4: METRICS */}
        {activeChapter === 'metrics' && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h1 className="text-5xl font-black mb-6 text-slate-900 tracking-tight">Thước đo <span className="text-blue-600">Hiệu năng</span></h1>
              <p className="text-xl text-slate-500 leading-relaxed">AI không chỉ có "Đúng" hay "Sai". Chúng ta cần những chỉ số tinh vi hơn để đánh giá sự tin cậy.</p>
            </section>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Accuracy */}
              <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-black">1</div>
                  <h3 className="text-xl font-black">Accuracy (Độ chính xác tổng)</h3>
                </div>
                <p className="text-sm text-slate-500 mb-6 italic leading-relaxed">"Tỷ lệ giữa số dự đoán đúng trên tổng số dự đoán."</p>
                <div className="bg-slate-50 p-4 rounded-2xl font-mono text-xs text-center border border-slate-100">
                  Accuracy = (TP + TN) / (TP + TN + FP + FN)
                </div>
              </div>

              {/* F1-Score */}
              <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center font-black">2</div>
                  <h3 className="text-xl font-black">F1-Score (Điểm cân bằng)</h3>
                </div>
                <p className="text-sm text-slate-500 mb-6 italic leading-relaxed">"Trung bình điều hòa giữa Precision và Recall. Đây là chỉ số quan trọng nhất khi dữ liệu bị lệch."</p>
                <div className="bg-slate-50 p-4 rounded-2xl font-mono text-xs text-center border border-slate-100">
                  F1 = 2 * (P * R) / (P + R)
                </div>
              </div>

              {/* Precision */}
              <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center font-black">3</div>
                  <h3 className="text-xl font-black">Precision (Độ chuẩn xác)</h3>
                </div>
                <p className="text-sm text-slate-500 mb-6 italic leading-relaxed">"Trong những bài tôi báo là Fake, có bao nhiêu bài thực sự là Fake?"</p>
                <div className="bg-slate-50 p-4 rounded-2xl font-mono text-xs text-center border border-slate-100">
                  Precision = TP / (TP + FP)
                </div>
              </div>

              {/* Recall */}
              <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-green-100 text-green-600 flex items-center justify-center font-black">4</div>
                  <h3 className="text-xl font-black">Recall (Độ phủ/Nhạy)</h3>
                </div>
                <p className="text-sm text-slate-500 mb-6 italic leading-relaxed">"Tôi đã bắt được bao nhiêu phần trăm trong tổng số tin giả thực tế?"</p>
                <div className="bg-slate-50 p-4 rounded-2xl font-mono text-xs text-center border border-slate-100">
                  Recall = TP / (TP + FN)
                </div>
              </div>
            </div>

            <div className="bg-red-900 p-10 rounded-[2.5rem] shadow-xl text-white">
              <h3 className="text-2xl font-black mb-4">⚠️ False Alarm Rate (Tỷ lệ báo động giả)</h3>
              <p className="text-red-100 leading-relaxed mb-6">
                Trong phát hiện tin giả, đây là chỉ số cực kỳ nguy hiểm. Nó cho biết tỷ lệ những **Tin thật bị AI gắn nhãn nhầm là Tin giả**. Nếu chỉ số này quá cao, AI sẽ trở nên "hoang tưởng" và đánh mất niềm tin của người dùng.
              </p>
              <div className="bg-white/10 p-4 rounded-2xl font-mono text-center border border-white/20">
                FAR = FP / (FP + TN)
              </div>
            </div>
          </div>
        )}

        {/* CHAPTER 5: CONFUSION MATRIX */}
        {activeChapter === 'matrix' && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h1 className="text-5xl font-black mb-6 text-slate-900 tracking-tight">Ma trận <span className="text-blue-600">Nhầm lẫn</span></h1>
              <p className="text-xl text-slate-500 leading-relaxed">Nơi mọi sự thật (và sai lầm) của AI được phơi bày rõ nét nhất.</p>
            </section>

            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-100">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div>
                   <div className="grid grid-cols-3 gap-3 text-center font-black uppercase text-[10px]">
                      <div></div>
                      <div className="text-slate-400 pb-2">Thực tế: REAL</div>
                      <div className="text-slate-400 pb-2">Thực tế: FAKE</div>
                      
                      <div className="text-slate-400 flex items-center justify-end pr-2">Dự đoán: REAL</div>
                      <div className="bg-green-100 p-8 rounded-2xl border-2 border-green-200 text-green-800 text-2xl">TN</div>
                      <div className="bg-red-50 p-8 rounded-2xl border-2 border-red-100 text-red-800 text-2xl">FN</div>

                      <div className="text-slate-400 flex items-center justify-end pr-2">Dự đoán: FAKE</div>
                      <div className="bg-red-50 p-8 rounded-2xl border-2 border-red-100 text-red-800 text-2xl">FP</div>
                      <div className="bg-green-100 p-8 rounded-2xl border-2 border-green-200 text-green-800 text-2xl">TP</div>
                   </div>
                </div>
                <div className="space-y-6">
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-green-600 text-white flex items-center justify-center shrink-0 font-bold">TP</div>
                    <div>
                      <h4 className="font-black text-slate-900">True Positive (Đúng Tuyệt đối)</h4>
                      <p className="text-sm text-slate-500">Tin giả và AI đã nhận diện đúng là tin giả. ✅</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center shrink-0 font-bold">TN</div>
                    <div>
                      <h4 className="font-black text-slate-900">True Negative (Thật Chuẩn xác)</h4>
                      <p className="text-sm text-slate-500">Tin thật và AI đã xác nhận đúng là tin thật. ✅</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-red-600 text-white flex items-center justify-center shrink-0 font-bold">FP</div>
                    <div>
                      <h4 className="font-black text-red-600">False Positive (Báo động giả)</h4>
                      <p className="text-sm text-slate-500">Tin thật nhưng AI lại bảo là tin giả. Đây là sự "oan sai". ❌</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-orange-600 text-white flex items-center justify-center shrink-0 font-bold">FN</div>
                    <div>
                      <h4 className="font-black text-orange-600">False Negative (Bỏ lọt tội phạm)</h4>
                      <p className="text-sm text-slate-500">Tin giả nhưng AI lại bảo là tin thật. Đây là sự "nguy hiểm". ❌</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TheoryPage;
