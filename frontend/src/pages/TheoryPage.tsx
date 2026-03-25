import { useState } from 'react';

// Import các chương lý thuyết từ thư mục con
import Chapter1NLP from './theory-chapters/Chapter1NLP';
import Chapter2Vector from './theory-chapters/Chapter2Vector';
import Chapter3Models from './theory-chapters/Chapter3Models';
import Chapter4Metrics from './theory-chapters/Chapter4Metrics';
import Chapter5Matrix from './theory-chapters/Chapter5Matrix';
import Chapter6Pipeline from './theory-chapters/Chapter6Pipeline';

const TheoryPage = () => {
  const [activeChapter, setActiveChapter] = useState('nlp');

  const chapters = [
    { id: 'nlp', title: '1. Tiền xử lý (NLP)', icon: '🧼' },
    { id: 'vector', title: '2. Vectơ hóa (TF-IDF & Doc2Vec)', icon: '🔢' },
    { id: 'models', title: '3. 6 Thuật toán Phân loại AI', icon: '🧠' },
    { id: 'metrics', title: '4. Chỉ số Đánh giá (Metrics)', icon: '📈' },
    { id: 'matrix', title: '5. Ma trận Nhầm lẫn (Confusion)', icon: '🧱' },
    { id: 'pipeline', title: '6. Luồng Xử Lý Hệ Thống', icon: '⚙️' },
  ];

  const renderContent = () => {
    switch (activeChapter) {
      case 'nlp': return <Chapter1NLP />;
      case 'vector': return <Chapter2Vector />;
      case 'models': return <Chapter3Models />;
      case 'metrics': return <Chapter4Metrics />;
      case 'matrix': return <Chapter5Matrix />;
      case 'pipeline': return <Chapter6Pipeline />;
      default: return <Chapter1NLP />;
    }
  };

  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-[#fcfcfd] text-slate-800 font-sans">
      {/* Sidebar */}
      <div className="w-full md:w-80 bg-white border-r border-slate-200 p-8 space-y-3 sticky top-0 h-fit md:h-screen overflow-y-auto shadow-sm z-20">
        <div className="mb-10 text-center md:text-left">
          <h2 className="text-2xl font-black text-slate-900 tracking-tighter uppercase leading-tight">
            AI Knowledge <br/><span className="text-blue-600">Base</span>
          </h2>
          <div className="h-1 w-12 bg-blue-600 mt-2 mx-auto md:mx-0"></div>
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

      {/* Main Content */}
      <div className="flex-1 p-6 md:p-16 max-w-5xl mx-auto overflow-y-auto w-full">
        {renderContent()}
      </div>
    </div>
  );
};

export default TheoryPage;
