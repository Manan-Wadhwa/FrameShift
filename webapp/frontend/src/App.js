import React, { useState } from 'react';
import { Upload, Settings, Play, AlertCircle } from 'lucide-react';

const FrameShiftApp = () => {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [image1Preview, setImage1Preview] = useState(null);
  const [image2Preview, setImage2Preview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [config, setConfig] = useState({
    use_roi: false,
    remove_background: true,
    use_edge_detection: true,
    filter_text_regions: true,
    sensitivity: 0.15,
  });

  const handleImageUpload = (e, imageNumber) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (imageNumber === 1) {
          setImage1(file);
          setImage1Preview(reader.result);
        } else {
          setImage2(file);
          setImage2Preview(reader.result);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!image1 || !image2) {
      setError('Please upload both images');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);
    formData.append('config', JSON.stringify(config));

    try {
      const API_URL = 'https://frameshift-api.onrender.com/analyze';
      
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to analyze images. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const ConfigToggle = ({ label, configKey, description }) => (
    <div className="flex items-start justify-between p-3 bg-gray-50 rounded-lg">
      <div className="flex-1">
        <label className="font-medium text-gray-700 flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={config[configKey]}
            onChange={(e) => setConfig({ ...config, [configKey]: e.target.checked })}
            className="mr-3 w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
          />
          {label}
        </label>
        <p className="text-sm text-gray-500 ml-7 mt-1">{description}</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">FrameShift</h1>
              <p className="text-gray-600 mt-1">Visual Comparison Engine for Time-Series Images</p>
            </div>
            <div className="bg-blue-100 px-4 py-2 rounded-lg">
              <span className="text-blue-800 font-semibold">v1.1</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
            <AlertCircle className="text-red-600 mr-3" size={20} />
            <span className="text-red-800">{error}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <Upload className="mr-2" size={24} />
                Upload Images
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 hover:border-blue-500 transition-colors">
                  <label className="cursor-pointer block">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => handleImageUpload(e, 1)}
                      className="hidden"
                    />
                    {image1Preview ? (
                      <div className="space-y-2">
                        <img src={image1Preview} alt="Preview 1" className="w-full h-48 object-cover rounded" />
                        <p className="text-sm text-green-600 font-medium text-center">✓ Image 1 Loaded</p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-48 text-gray-400">
                        <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <p className="mt-2 text-sm font-medium">Upload Image 1 (Before)</p>
                        <p className="text-xs mt-1">Click or drag to upload</p>
                      </div>
                    )}
                  </label>
                </div>

                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 hover:border-blue-500 transition-colors">
                  <label className="cursor-pointer block">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => handleImageUpload(e, 2)}
                      className="hidden"
                    />
                    {image2Preview ? (
                      <div className="space-y-2">
                        <img src={image2Preview} alt="Preview 2" className="w-full h-48 object-cover rounded" />
                        <p className="text-sm text-green-600 font-medium text-center">✓ Image 2 Loaded</p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-48 text-gray-400">
                        <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <p className="mt-2 text-sm font-medium">Upload Image 2 (After)</p>
                        <p className="text-xs mt-1">Click or drag to upload</p>
                      </div>
                    )}
                  </label>
                </div>
              </div>
            </div>

            {results && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Analysis Results</h2>
                
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Changes Detected</p>
                    <p className="text-3xl font-bold text-blue-600">{results.changes_count}</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">SSIM Score</p>
                    <p className="text-3xl font-bold text-green-600">{results.ssim_score?.toFixed(3)}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.difference_map && (
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-2">Difference Map</p>
                      <img 
                        src={`data:image/png;base64,${results.difference_map}`} 
                        alt="Difference Map" 
                        className="w-full rounded-lg border"
                      />
                    </div>
                  )}
                  {results.annotated_image && (
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-2">Detected Changes</p>
                      <img 
                        src={`data:image/png;base64,${results.annotated_image}`} 
                        alt="Annotated" 
                        className="w-full rounded-lg border"
                      />
                    </div>
                  )}
                </div>

                {results.changes && results.changes.length > 0 && (
                  <div className="mt-6">
                    <h3 className="font-semibold text-gray-800 mb-3">Top Changes:</h3>
                    <div className="space-y-2">
                      {results.changes.slice(0, 5).map((change, idx) => (
                        <div key={idx} className="bg-gray-50 rounded p-3 text-sm">
                          <span className="font-medium text-blue-600">#{idx + 1}</span>
                          <span className="ml-3 text-gray-700">
                            Location: ({change.bbox[0]}, {change.bbox[1]}) • 
                            Size: {change.area.toFixed(0)} px²
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <Settings className="mr-2" size={24} />
                Configuration
              </h2>

              <div className="space-y-3">
                <ConfigToggle
                  label="ROI Selection"
                  configKey="use_roi"
                  description="Select specific region for analysis"
                />
                
                <ConfigToggle
                  label="Background Removal"
                  configKey="remove_background"
                  description="Remove background before comparison"
                />
                
                <ConfigToggle
                  label="Edge Detection"
                  configKey="use_edge_detection"
                  description="Detect texture changes (tire wear, etc.)"
                />
                
                <ConfigToggle
                  label="Filter Text Regions"
                  configKey="filter_text_regions"
                  description="Ignore text labels and overlays"
                />

                <div className="p-3 bg-gray-50 rounded-lg">
                  <label className="font-medium text-gray-700 block mb-2">
                    Sensitivity: {config.sensitivity.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.30"
                    step="0.01"
                    value={config.sensitivity}
                    onChange={(e) => setConfig({ ...config, sensitivity: parseFloat(e.target.value) })}
                    className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>More sensitive</span>
                    <span>Less sensitive</span>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!image1 || !image2 || loading}
              className={`w-full py-4 rounded-xl font-semibold text-white transition-all flex items-center justify-center ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : image1 && image2
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg hover:shadow-xl'
                  : 'bg-gray-300 cursor-not-allowed'
              }`}
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="mr-2" size={20} />
                  Analyze Images
                </>
              )}
            </button>

            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
              <h3 className="font-bold text-gray-800 mb-3">💡 Quick Guide</h3>
              <div className="space-y-3 text-sm text-gray-700">
                <div>
                  <p className="font-semibold text-blue-700">🏎️ For F1 Tire Wear:</p>
                  <p className="text-xs mt-1">Enable Edge Detection + Background Removal</p>
                </div>
                <div>
                  <p className="font-semibold text-blue-700">🎯 For Different POV:</p>
                  <p className="text-xs mt-1">Enable ROI Selection + Background Removal</p>
                </div>
                <div>
                  <p className="font-semibold text-blue-700">📝 For Broadcast Footage:</p>
                  <p className="text-xs mt-1">Enable Filter Text Regions</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="bg-white border-t mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>FrameShift v1.1 - Visual Comparison Engine for Time-Series Images</p>
          <p className="mt-1">Built for F1 Analysis • Tire Degradation • Position Changes • Incident Detection</p>
        </div>
      </footer>
    </div>
  );
};

export default FrameShiftApp;