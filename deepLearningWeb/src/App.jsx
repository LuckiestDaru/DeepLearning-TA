import { useState, useEffect } from 'react';

export default function App() {
  // --- STATE MANAGEMENT ---
  const [inputType, setInputType] = useState('none'); // 'none' | 'webcam' | 'upload'
  const [imageResult, setImageResult] = useState(null); // Untuk hasil gambar statis
  const [stats, setStats] = useState({ total: 0, car: 0, motorbike: 0, bus: 0, truck: 0 });
  const [streamTrigger, setStreamTrigger] = useState(0); // Pemicu refresh video

  // Konfigurasi Pilihan User
  const [config, setConfig] = useState({
    mode: 'detect',
    enhancement: 'none',
    direction: 'top-down'
  });

  // --- USE EFFECT (STATS LOOP) ---
  useEffect(() => {
    const interval = setInterval(() => {
      // Hanya fetch stats jika sistem aktif (bukan none)
      if (inputType !== 'none') {
        fetch('http://localhost:5000/stats')
          .then(res => res.json())
          .then(data => setStats(data))
          .catch(err => console.error("Backend offline?", err));
      }
    }, 1000); // Update setiap 1 detik
    return () => clearInterval(interval);
  }, [inputType]);

  // --- API HANDLERS ---

  const updateConfig = async (newConfig) => {
    const updated = { ...config, ...newConfig };
    setConfig(updated);
    await fetch('http://localhost:5000/update_config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    });
    
    // Refresh gambar statis jika user ganti filter saat mode upload gambar
    if (inputType === 'upload' && imageResult) {
       fetchImageResult();
    }
  };

  const handleSetNone = async () => {
    setInputType('none');
    setImageResult(null);
    await fetch('http://localhost:5000/stop_camera', { method: 'POST' });
  };

  const handleSetWebcam = async () => {
    setInputType('webcam');
    setImageResult(null);
    await fetch('http://localhost:5000/set_webcam', { method: 'POST' });
    setStreamTrigger(prev => prev + 1); // Paksa refresh video player
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setInputType('upload'); // Set UI ke mode upload
      
      const res = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.type === 'image') {
        fetchImageResult(); // Ambil hasil gambar
      } else {
        setImageResult(null); // Mode video
        setStreamTrigger(prev => prev + 1); // Paksa refresh video player
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Gagal upload file. Pastikan backend menyala.");
    }
  };

  const fetchImageResult = async () => {
    try {
      const res = await fetch('http://localhost:5000/processed_image');
      const data = await res.json();
      if (data.image) setImageResult(data.image);
    } catch (e) { console.error(e); }
  };

  // --- RENDER UI ---

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4 md:p-8 font-sans">
      <header className="mb-8 text-center">
        <h1 className="text-3xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-2">
          Deep Learning Vehicle Analysis
        </h1>
        <p className="text-gray-400 text-sm md:text-base">YOLOv11 Detection & Directional Counting System</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto">
        
        {/* --- LEFT SIDEBAR (CONTROLS) --- */}
        <div className="lg:col-span-4 space-y-6">
          
          <div className="bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
            <h2 className="text-xl font-bold mb-6 border-b border-slate-600 pb-3 text-white">
              ⚙️ Control Panel
            </h2>

            {/* 1. INPUT SOURCE */}
            <div className="mb-6">
              <label className="text-xs font-bold text-gray-400 uppercase mb-2 block tracking-wider">Input Source</label>
              <div className="grid grid-cols-3 gap-2">
                <button 
                  onClick={handleSetNone}
                  className={`py-2 rounded-lg text-xs font-bold transition-all duration-200 border border-transparent
                    ${inputType === 'none' 
                      ? 'bg-red-500/20 text-red-400 border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]' 
                      : 'bg-slate-700 hover:bg-slate-600 text-gray-300'}`}>
                  NONE
                </button>
                <button 
                  onClick={handleSetWebcam}
                  className={`py-2 rounded-lg text-xs font-bold transition-all duration-200 border border-transparent
                    ${inputType === 'webcam' 
                      ? 'bg-blue-500/20 text-blue-400 border-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.3)]' 
                      : 'bg-slate-700 hover:bg-slate-600 text-gray-300'}`}>
                  WEBCAM
                </button>
                <button 
                  onClick={() => document.getElementById('fileInput').click()} 
                  className={`py-2 rounded-lg text-xs font-bold transition-all duration-200 border border-transparent
                    ${inputType === 'upload' 
                      ? 'bg-green-500/20 text-green-400 border-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]' 
                      : 'bg-slate-700 hover:bg-slate-600 text-gray-300'}`}>
                  UPLOAD
                </button>
              </div>
              
              <input 
                id="fileInput"
                type="file" 
                onChange={handleFileUpload}
                className="hidden"
              />
              {inputType === 'upload' && (
                <div className="mt-2 text-center text-xs text-green-400 animate-pulse">
                  • File Active
                </div>
              )}
            </div>

            {/* 2. MODE & DIRECTION */}
            <div className="space-y-4 mb-6">
              <div>
                <label className="text-xs font-bold text-gray-400 uppercase mb-2 block tracking-wider">Operation Mode</label>
                <select 
                  value={config.mode}
                  onChange={(e) => updateConfig({ mode: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg p-2.5 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all">
                  <option value="detect">🔍 Detection Only</option>
                  <option value="count-video">📈 Counting & Tracking</option>
                </select>
              </div>

              {/* Conditional Rendering: Direction Dropdown */}
              {config.mode === 'count-video' && (
                <div className="animate-fade-in pt-2">
                  <label className="text-xs font-bold text-yellow-500 uppercase mb-2 block tracking-wider">
                    Counting Direction
                  </label>
                  <select 
                    value={config.direction}
                    onChange={(e) => updateConfig({ direction: e.target.value })}
                    className="w-full bg-slate-900 border border-yellow-500/30 rounded-lg p-2.5 text-sm text-yellow-100 focus:ring-2 focus:ring-yellow-500 outline-none transition-all">
                    <option value="top-down">⬇️ Atas ke Bawah (Top-Down)</option>
                    <option value="left-right">➡️ Kiri Kanan (Horizontal)</option>
                  </select>
                  <p className="text-[10px] text-gray-500 mt-1">
                    *Garis akan menyesuaikan arah (Horizontal/Vertikal)
                  </p>
                </div>
              )}
            </div>

            {/* 3. ENHANCEMENT */}
            <div className="mb-2">
                <label className="text-xs font-bold text-gray-400 uppercase mb-2 block tracking-wider">Image Filter</label>
                <div className="grid grid-cols-3 gap-2">
                  {['none', 'contrast', 'brightness'].map(type => (
                    <button
                      key={type}
                      onClick={() => updateConfig({ enhancement: type })}
                      className={`p-2 rounded-lg text-xs capitalize transition-all border border-transparent
                        ${config.enhancement === type 
                          ? 'bg-purple-500/20 text-purple-400 border-purple-500 font-bold' 
                          : 'bg-slate-700 hover:bg-slate-600 text-gray-400'}`}>
                      {type}
                    </button>
                  ))}
                </div>
            </div>
          </div>

          {/* 4. LIVE STATS */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-2xl shadow-xl border border-slate-700">
            <h3 className="text-green-400 font-bold mb-4 flex items-center gap-2 border-b border-slate-700 pb-2">
               📊 Live Statistics
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">🚗 Car</span> 
                <span className="font-mono bg-slate-700 px-2 py-1 rounded text-blue-300">{stats.car}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">🏍️ Motorbike</span> 
                <span className="font-mono bg-slate-700 px-2 py-1 rounded text-blue-300">{stats.motorbike}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">🚌 Bus</span> 
                <span className="font-mono bg-slate-700 px-2 py-1 rounded text-blue-300">{stats.bus}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">🚚 Truck</span> 
                <span className="font-mono bg-slate-700 px-2 py-1 rounded text-blue-300">{stats.truck}</span>
              </div>
              <div className="border-t border-slate-600 pt-3 mt-2 flex justify-between font-bold text-xl text-white">
                <span>TOTAL</span> <span className="text-green-400">{stats.total}</span>
              </div>
            </div>
          </div>

        </div>

        {/* --- MAIN DISPLAY (VIDEO PLAYER) --- */}
        <div className="lg:col-span-8">
          <div className="bg-black rounded-2xl overflow-hidden shadow-2xl border border-slate-800 relative aspect-video flex items-center justify-center group">
            
            {/* LOGIC TAMPILAN */}
            {inputType === 'none' ? (
               <div className="text-center text-gray-600 flex flex-col items-center animate-pulse">
                  <svg className="w-20 h-20 mb-4 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                  <p className="text-xl font-semibold">Camera Inactive</p>
                  <p className="text-sm mt-1">Select Webcam or Upload to start detection</p>
               </div>
            ) : (
              <>
                {imageResult ? (
                  <img src={imageResult} alt="Processed Static" className="w-full h-full object-contain" />
                ) : (
                  <img 
                    // Trik ?t=streamTrigger memaksa browser reload gambar
                    src={`http://localhost:5000/video_feed?t=${streamTrigger}`} 
                    alt="Live Stream" 
                    className="w-full h-full object-contain bg-black"
                  />
                )}
              </>
            )}

            {/* Overlay Status Badge */}
            <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-4 py-1.5 rounded-full border border-white/10 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${inputType === 'none' ? 'bg-red-500' : 'bg-green-500 animate-pulse'}`}></div>
              <span className="text-xs font-mono font-bold text-white tracking-wide">
                {inputType === 'none' ? 'OFFLINE' : inputType.toUpperCase()}
              </span>
            </div>

          </div>
          
          <div className="mt-4 text-center text-xs text-gray-500">
            Backend Processing: YOLOv11m • Resolution: Dynamic • Inference: GPU/CPU
          </div>
        </div>

      </div>
    </div>
  );
}