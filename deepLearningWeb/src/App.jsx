import { useState, useEffect } from 'react';

//ambil dari .env atau default ke localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function App() {
  //states
  const [inputType, setInputType] = useState('none'); 
  const [imageResult, setImageResult] = useState(null); 
  const [stats, setStats] = useState({ total: 0, car: 0, motorbike: 0, bus: 0, truck: 0 });
  const [streamTrigger, setStreamTrigger] = useState(0); 
  const [isImageMode, setIsImageMode] = useState(false); 

  const [config, setConfig] = useState({
    mode: 'detect',
    enhancement: 'none',
    direction: 'top-down'
  });

  // efekt untuk polling stats setiap detik saat input aktif
  useEffect(() => {
    const interval = setInterval(() => {
      if (inputType !== 'none') {
        fetch(`${API_BASE}/stats`)
          .then(res => res.json())
          .then(data => setStats(data))
          .catch(err => console.error("Backend offline?", err));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [inputType]);

  //handlers
  const updateConfig = async (newConfig) => {
    const updated = { ...config, ...newConfig };
    setConfig(updated);
    await fetch(`${API_BASE}/update_config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    });
    if (inputType === 'upload' && imageResult) fetchImageResult();
  };

  const handleSetNone = async () => {
    setInputType('none');
    setImageResult(null);
    setIsImageMode(false);
    await fetch(`${API_BASE}/stop_camera`, { method: 'POST' });
  };

  const handleSetWebcam = async () => {
    setInputType('webcam');
    setImageResult(null);
    setIsImageMode(false);
    await fetch(`${API_BASE}/set_webcam`, { method: 'POST' });
    setStreamTrigger(prev => prev + 1);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      setInputType('upload');
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.type === 'image') {
        setIsImageMode(true);
        fetchImageResult();
      } else {
        setIsImageMode(false);
        setImageResult(null);
        setStreamTrigger(prev => prev + 1);
      }
    } catch (error) {
      alert("Gagal upload file.");
    }
  };

  const fetchImageResult = async () => {
    try {
      const res = await fetch(`${API_BASE}/processed_image`);
      const data = await res.json();
      if (data.image) {
        setImageResult(data.image);
        // refresh stats
        fetch(`${API_BASE}/stats`)
          .then(res => res.json())
          .then(data => setStats(data));
      }
    } catch (e) { console.error(e); }
  };

  return (
    // Update Layout: flex-col (Mobile: Atas-Bawah) -> md:flex-row (Desktop: Kiri-Kanan)
    <div className="flex flex-col md:flex-row h-screen w-screen bg-[#1e1e1e] text-gray-300 font-sans overflow-hidden">
      
      {/* Sidebar: 
          - Order 2 (Mobile: Di Bawah) -> md:Order 1 (Desktop: Di Kiri)
          - Width Full (Mobile) -> md:w-72 (Desktop)
          - Height 45% (Mobile) -> md:h-full (Desktop)
          - Border Top (Mobile) -> Border Right (Desktop)
      */}
      <aside className="order-2 md:order-1 w-full md:w-72 h-[45%] md:h-full bg-[#1e1e1e] border-t md:border-t-0 md:border-r border-[#333] flex flex-col flex-shrink-0 z-20">
        
        {/* Header */}
        <div className="h-12 md:h-16 flex items-center justify-center border-b border-[#333] flex-shrink-0">
          <h1 className="font-bold text-white tracking-widest text-sm">VEHICLE DETECTION</h1>
        </div>

        {/* Konten Sidebar */}
        <div className="flex-1 overflow-y-auto p-4 md:p-5 space-y-6 md:space-y-8 custom-scrollbar">
          
          {/* Source Controls */}
          <div>
            <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-3 block">Input Source</label>
            <div className="flex flex-col gap-2">
              <button onClick={handleSetNone} 
                className={`w-full text-left px-4 py-3 rounded text-xs font-bold transition-all border ${inputType === 'none' ? 'bg-red-900/20 border-red-800 text-red-400' : 'bg-[#252525] border-transparent text-gray-400 hover:bg-[#303030]'}`}>
                ⛔ STOP
              </button>
              
              <div className="grid grid-cols-2 gap-2">
                <button onClick={handleSetWebcam} 
                  className={`px-3 py-3 rounded text-xs font-bold transition-all border ${inputType === 'webcam' ? 'bg-green-900/20 border-green-800 text-green-400' : 'bg-[#252525] border-transparent text-gray-400 hover:bg-[#303030]'}`}>
                  WEBCAM
                </button>
                <button onClick={() => document.getElementById('fileInput').click()} 
                  className={`px-3 py-3 rounded text-xs font-bold transition-all border ${inputType === 'upload' ? 'bg-green-900/20 border-green-800 text-green-400' : 'bg-[#252525] border-transparent text-gray-400 hover:bg-[#303030]'}`}>
                  UPLOAD
                </button>
              </div>
              <input id="fileInput" type="file" onChange={handleFileUpload} className="hidden" />
            </div>
          </div>

          {/* Settings */}
          <div className="space-y-6">
            <div>
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2 block">Detection Mode</label>
              <select value={config.mode} onChange={(e) => updateConfig({ mode: e.target.value })} 
                className="w-full bg-[#252525] border border-[#333] text-gray-200 text-xs rounded p-2.5 outline-none focus:border-gray-500">
                <option value="detect">Standard Detection</option>
                <option value="count-video">Counting & Tracking (video)</option>
              </select>
            </div>

            {config.mode === 'count-video' && !isImageMode && (
              <div>
                <label className="text-[10px] font-bold text-yellow-600 uppercase tracking-widest mb-2 block">Direction</label>
                <select value={config.direction} onChange={(e) => updateConfig({ direction: e.target.value })} 
                  className="w-full bg-[#252525] border border-yellow-800/30 text-yellow-500 text-xs rounded p-2.5 outline-none focus:border-yellow-600">
                  <option value="top-down">Top - Down</option>
                  <option value="left-right">Left - Right</option>
                </select>
              </div>
            )}

            <div>
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2 block">Filter</label>
              <div className="flex bg-[#252525] p-1 rounded">
                {['none', 'contrast', 'brightness'].map(type => (
                  <button key={type} onClick={() => updateConfig({ enhancement: type })}
                    className={`flex-1 py-1.5 rounded text-[10px] font-bold uppercase transition-all ${config.enhancement === type ? 'bg-[#333] text-white shadow' : 'text-gray-500 hover:text-gray-300'}`}>
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats Table */}
        <div className="p-4 md:p-5 bg-[#1a1a1a] border-t border-[#333] flex-shrink-0">
           <div className="space-y-2">
             {[
               { l: 'Car', v: stats.car },
               { l: 'Motorbike', v: stats.motorbike },
               { l: 'Bus', v: stats.bus },
               { l: 'Truck', v: stats.truck },
             ].map(item => (
               <div key={item.l} className="flex justify-between items-center text-xs">
                 <span className="text-gray-500">{item.l}</span>
                 <span className="text-white font-mono">{item.v}</span>
               </div>
             ))}
             <div className="h-px bg-[#333] my-2"></div>
             <div className="flex justify-between items-center">
               <span className="text-xs font-bold text-gray-400">TOTAL</span>
               <span className="text-lg font-bold text-green-500 font-mono">{stats.total}</span>
             </div>
          </div>
        </div>
      </aside>
      
      <main className="order-1 md:order-2 flex-1 flex flex-col bg-black relative overflow-hidden">
        
        {/* Status Overlay */}
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1 rounded bg-black/50 backdrop-blur border border-white/10">
           <div className={`w-2 h-2 rounded-full ${inputType === 'none' ? 'bg-red-500' : 'bg-green-500 animate-pulse'}`}></div>
           <span className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">
             {inputType === 'none' ? 'OFFLINE' : 'LIVE FEED'}
           </span>
        </div>

        {/* Video Container */}
        <div className="w-full h-full flex items-center justify-center">
            {inputType === 'none' ? (
              <div className="text-center opacity-40">
                <div className="w-16 h-16 border-2 border-gray-600 rounded-full flex items-center justify-center mx-auto mb-4">
                   <div className="w-2 h-2 bg-gray-600 rounded-full animate-ping"></div>
                </div>
                <p className="text-gray-500 text-xs tracking-[0.2em] uppercase">Ready for Input Detecting</p>
              </div>
            ) : (
              <>
                 {imageResult ? (
                    <img src={imageResult} alt="Result" className="w-full h-full object-contain" />
                 ) : (
                    <img 
                      src={`${API_BASE}/video_feed?t=${streamTrigger}`} 
                      alt="Stream" 
                      className="w-full h-full object-contain" 
                    />
                 )}
                 
                 {/* Count Overlay */}
                 {isImageMode && (
                   <div className="absolute bottom-6 left-6 z-10 bg-black/70 backdrop-blur px-4 py-2 rounded border-l-4 border-green-500">
                      <p className="text-[10px] text-gray-400 uppercase tracking-wider">Detected</p>
                      <p className="text-2xl font-mono text-white">{stats.total}</p>
                   </div>
                 )}
              </>
            )}
        </div>
      </main>
    </div>
  );
}