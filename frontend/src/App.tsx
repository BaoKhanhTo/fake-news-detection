import React, { useState, useEffect } from 'react'
import axios from 'axios'
import TheoryPage from './pages/TheoryPage'

interface ModelEducation {
  name: string;
  concept: string;
  principle: string;
  flow: string;
}

interface ModelMetrics {
  accuracy: number;
  f1: number;
  recall: number;
  precision: number;
  false_alarm_rate: number;
  confusion_matrix: number[][];
}

interface ModelResult {
  prediction: string;
  fake_probability: number;
  real_probability: number;
  metrics: ModelMetrics;
  education: ModelEducation;
}

interface PredictionResponse {
  input_text: string;
  cleaned_text: string;
  results: Record<string, ModelResult>;
  doc2vec_vector: number[];
  education_doc2vec: ModelEducation;
}

function App() {
  const [currentView, setCurrentView] = useState<'detector' | 'theory'>('detector')
  const [newsText, setNewsText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<string>('')

  useEffect(() => {
    if (result && result.results && !activeTab) {
      const firstModel = Object.keys(result.results)[0];
      if (firstModel) setActiveTab(firstModel);
    }
  }, [result])

  const handleCheck = async () => {
    if (!newsText.trim()) {
      setError('Vui lòng nhập nội dung tin tức cần kiểm tra.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setActiveTab('')

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        text: newsText
      })
      if (response.data && response.data.results && Object.keys(response.data.results).length > 0) {
        setResult(response.data)
      } else {
        setError('Máy chủ không trả về kết quả phân tích.')
      }
    } catch (err: any) {
      console.error('API Error:', err)
      setError(`Lỗi kết nối máy chủ: ${err.response?.data?.detail || err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const renderMetrics = (metrics?: ModelMetrics) => {
    if (!metrics) return null;
    return (
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        {[
          { label: 'Độ chính xác', val: metrics.accuracy, color: 'blue' },
          { label: 'F1-Score', val: metrics.f1, color: 'purple' },
          { label: 'Recall', val: metrics.recall, color: 'green' },
          { label: 'Precision', val: metrics.precision, color: 'orange' },
          { label: 'Báo động giả', val: metrics.false_alarm_rate, color: 'red' },
        ].map((m, i) => (
          <div key={i} className={`bg-${m.color}-50 p-3 rounded-lg text-center border border-${m.color}-100 shadow-sm`}>
            <p className={`text-[10px] text-${m.color}-600 font-black uppercase mb-1`}>{m.label}</p>
            <p className={`text-xl font-black text-${m.color}-800`}>{(m.val * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    )
  }

  if (currentView === 'theory') {
    return (
      <>
        <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 px-6 py-4">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <div className="font-black text-xl text-slate-900 tracking-tighter">AI <span className="text-blue-600">ACADEMY</span></div>
            <button 
              onClick={() => setCurrentView('detector')}
              className="px-6 py-2 bg-slate-900 text-white rounded-full text-sm font-bold hover:bg-slate-800 transition-all"
            >
              ← Quay lại Kiểm tra
            </button>
          </div>
        </nav>
        <TheoryPage />
      </>
    )
  }

  return (
    <div className="min-h-screen bg-[#f8fafc] font-sans">
      {/* Header with Nav */}
      <nav className="bg-white border-b border-slate-100 px-6 py-4 mb-12 shadow-sm">
        <div className="max-w-5xl mx-auto flex justify-between items-center">
          <div className="font-black text-2xl text-slate-900 tracking-tighter">FAKE<span className="text-blue-600">NEWS</span> DETECT</div>
          <button 
            onClick={() => setCurrentView('theory')}
            className="flex items-center gap-2 px-6 py-2 bg-blue-50 text-blue-600 rounded-full text-sm font-black hover:bg-blue-100 transition-all border border-blue-200"
          >
            📚 HỌC LÝ THUYẾT AI
          </button>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-4 pb-20">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-black text-slate-900 mb-4 tracking-tight">Hệ Thống Phân Tích <span className="text-blue-600">Đa Mô Hình</span></h1>
          <p className="text-lg text-slate-500 max-w-2xl mx-auto font-medium">Nhập nội dung tin tức để xem cách các thuật toán AI khác nhau đưa ra kết luận.</p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-3xl shadow-2xl p-8 mb-12 border border-slate-100">
          <textarea
            rows={6}
            className="block w-full rounded-2xl border-2 border-slate-100 shadow-sm focus:border-blue-500 focus:ring-4 focus:ring-blue-50/50 text-slate-700 p-5 border outline-none transition-all duration-300 text-base"
            placeholder="Dán nội dung bài báo vào đây..."
            value={newsText}
            onChange={(e) => setNewsText(e.target.value)}
          />
          
          {error && (
            <div className="mt-4 p-4 rounded-xl bg-red-50 text-red-600 text-sm font-bold border border-red-100 animate-pulse">
              ⚠️ {error}
            </div>
          )}

          <div className="mt-8 flex justify-center">
            <button
              onClick={handleCheck}
              disabled={loading}
              className={`px-12 py-4 text-lg font-black rounded-2xl shadow-xl text-white transition-all duration-300 ${
                loading ? 'bg-slate-400 cursor-not-allowed scale-95' : 'bg-blue-600 hover:bg-blue-700 hover:scale-105 active:scale-95'
              }`}
            >
              {loading ? 'Đang phân tích...' : 'Bắt đầu kiểm chứng'}
            </button>
          </div>
        </div>

        {/* Results... (giữ nguyên logic kết quả nhưng trình bày gọn hơn) */}
        {result && result.results && (
          <div className="animate-in fade-in duration-700">
             {/* Model Tabs */}
             <div className="flex flex-wrap gap-2 mb-8 justify-center">
              {Object.keys(result.results).map((name) => (
                <button
                  key={name}
                  onClick={() => setActiveTab(name)}
                  className={`px-6 py-3 rounded-2xl text-xs font-black border-2 transition-all uppercase tracking-widest ${
                    activeTab === name 
                      ? 'bg-slate-900 text-white border-slate-900 shadow-xl scale-110' 
                      : 'bg-white text-slate-400 border-slate-100 hover:border-slate-300'
                  }`}
                >
                  {result.results[name].education?.name?.split(' (')[0] || name}
                </button>
              ))}
              <button
                onClick={() => setActiveTab('doc2vec')}
                className={`px-6 py-3 rounded-2xl text-xs font-black border-2 transition-all uppercase tracking-widest ${
                  activeTab === 'doc2vec' 
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-xl scale-110' 
                    : 'bg-white text-slate-400 border-slate-100 hover:border-indigo-300'
                }`}
              >
                Doc2Vec Vector
              </button>
            </div>

            {activeTab && activeTab !== 'doc2vec' && result.results[activeTab] && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                  <div className={`rounded-3xl shadow-xl p-8 border-t-8 bg-white ${
                    result.results[activeTab].prediction === 'Fake' ? 'border-red-500' : 'border-green-500'
                  }`}>
                    <div className="flex items-center justify-between mb-8">
                      <h2 className={`text-5xl font-black ${
                        result.results[activeTab].prediction === 'Fake' ? 'text-red-600' : 'text-green-600'
                      }`}>
                        {result.results[activeTab].prediction === 'Fake' ? 'TIN GIẢ' : 'TIN THẬT'}
                      </h2>
                      <div className="text-right">
                        <span className="text-[10px] font-black text-slate-400 uppercase block mb-1">Xác suất giả</span>
                        <p className="text-3xl font-black text-slate-800">
                          {((result.results[activeTab].fake_probability || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    {renderMetrics(result.results[activeTab].metrics)}
                  </div>

                  <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
                    <h3 className="text-sm font-black text-slate-400 uppercase mb-6 tracking-widest">💡 Giải thích từ AI</h3>
                    <p className="text-slate-700 font-bold mb-4 text-lg">{result.results[activeTab].education?.concept}</p>
                    <p className="text-slate-500 italic leading-relaxed">{result.results[activeTab].education?.principle}</p>
                  </div>
                </div>

                <div className="bg-slate-900 rounded-3xl p-8 text-white shadow-2xl h-fit sticky top-24">
                  <h3 className="text-xs font-black text-blue-400 uppercase mb-6 tracking-widest">Quy trình xử lý</h3>
                  <div className="space-y-4">
                    {result.results[activeTab].education?.flow?.split(' -> ').map((step, i) => (
                      <div key={i} className="flex items-center gap-4">
                        <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-[10px] font-bold">{i+1}</div>
                        <div className="text-sm font-bold text-slate-300">{step}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'doc2vec' && (
              <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100 animate-in zoom-in duration-500">
                <h2 className="text-3xl font-black text-indigo-700 mb-4 uppercase">Đặc trưng ngữ nghĩa (Doc2Vec)</h2>
                <p className="text-slate-500 mb-8 font-medium">Văn bản của bạn đã được chuyển đổi thành dãy số 100 chiều để AI so sánh với kho dữ liệu khổng lồ.</p>
                <div className="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-10 gap-2">
                  {result.doc2vec_vector?.map((val, i) => (
                    <div key={i} className="flex flex-col items-center">
                      <div className="w-full bg-indigo-50 border border-indigo-100 p-2 text-center text-[10px] font-bold text-indigo-600 rounded-lg">
                        {val.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
