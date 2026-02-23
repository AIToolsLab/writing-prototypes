
import { useState, useEffect } from 'react';
import { getHighlights, Highlight } from '../api';
import { Sliders, RefreshCw, AlertCircle, ArrowRight } from 'lucide-react';

const DEFAULT_DOC =
  "Deep learning neural network technology advances are pretty cool if you are careful to use it in ways that don't take stuff from people.";

const PROMPT_OPTIONS = [
  "Rewrite this document to be more clear and concise.",
  "Rewrite this document to be more detailed and engaging.",
  "Summarize this document in one sentence.",
  "Translate this document into Spanish.",
];

export default function Highlights() {
  const [prompt, setPrompt] = useState(PROMPT_OPTIONS[0]);
  const [doc, setDoc] = useState(DEFAULT_DOC);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [loading, setLoading] = useState(false);
  const [minLoss, setMinLoss] = useState(0);

  const fetchHighlights = async () => {
    if (!doc.trim()) return;
    setLoading(true);
    try {
      const result = await getHighlights(prompt, doc, doc);
      setHighlights(result);
      
      // Calculate loss statistics
      if (result.length > 0) {
         const losses = result.slice(1).map(h => h.token_loss);
         const highestLoss = Math.max(...losses);
         // Heuristic: start at 50% of peak loss if minLoss is 0 (initial load)
         if (minLoss === 0) setMinLoss(highestLoss * 0.5);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (doc) fetchHighlights();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Initial load only

  return (
    <div className="max-w-5xl mx-auto p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-gray-900">Highlight Possible Edits</h2>
        <div className="flex space-x-2">
             <button 
                onClick={fetchHighlights}
                disabled={loading}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
                <RefreshCw size={16} className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
                Analyze Text
            </button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Controls Column */}
        <div className="md:col-span-1 space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 space-y-4">
                <h3 className="font-semibold text-gray-900 flex items-center">
                    <Sliders size={18} className="mr-2" /> Configuration
                </h3>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Prompt</label>
                    <select 
                        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        value={prompt} 
                        onChange={e => setPrompt(e.target.value)}
                    >
                    {PROMPT_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Input Text</label>
                    <textarea 
                        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[150px]"
                        value={doc} 
                        onChange={e => setDoc(e.target.value)}
                        placeholder="Enter text to analyze..."
                    />
                </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 space-y-4">
                 <h3 className="font-semibold text-gray-900 flex items-center">
                    <AlertCircle size={18} className="mr-2" /> Sensitivity
                </h3>
                <div className="space-y-6">
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm text-gray-600">Threshold (Loss)</span>
                            <span className="text-sm font-medium text-blue-600">{minLoss.toFixed(2)}</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" 
                            max="15" 
                            step="0.1" 
                            value={minLoss} 
                            onChange={e => setMinLoss(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        />
                        <p className="text-xs text-gray-500 mt-2">
                            Lower values highlight more tokens (fewer "surprises"). Higher values highlight only the most unexpected tokens.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        {/* Visualization Column */}
        <div className="md:col-span-2">
             <div className="bg-white rounded-xl shadow-sm border border-gray-200 min-h-[500px] flex flex-col">
                <div className="border-b border-gray-100 p-4 bg-gray-50 rounded-t-xl flex justify-between items-center">
                    <span className="font-semibold text-gray-700">Analysis Result</span>
                    {highlights.length > 0 && (
                        <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded border border-gray-200">
                            {highlights.filter(h => h.token_loss > minLoss).length} highlighted tokens
                        </span>
                    )}
                </div>
                
                <div className="p-6 md:p-8 leading-loose text-lg text-gray-800 font-serif">
                     {loading ? (
                        <div className="flex items-center justify-center h-full text-gray-400">
                             <span className="animate-pulse">Analyzing text patterns...</span>
                        </div>
                     ) : highlights.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-3">
                            <ArrowRight size={32} className="text-gray-300" />
                            <p>Enter text and click Analyze to categorize tokens.</p>
                        </div>
                     ) : (
                        <div className="highlights-container">
                        {highlights.map((h, i) => {
                        const isHighlight = h.token !== h.most_likely_token && (h.token_loss > minLoss); 
                        // Determine intensity of highlight based on loss difference? 
                        // For now keep binary but clean.
                        return (
                            <span 
                            key={i} 
                            className={`relative inline-block px-[1px] mx-[1px] rounded transition-colors duration-200 ${
                                isHighlight 
                                ? 'bg-red-50 border-b-2 border-red-300 cursor-help hover:bg-red-100' 
                                : 'hover:bg-gray-50'
                            }`}
                            title={`Actual: "${h.token.trim()}" | Predicted: "${h.most_likely_token}" | Loss: ${h.token_loss.toFixed(2)}`}
                            >
                            {h.token === '\n' ? <br /> : h.token.replace(/\n/g, ' ')}
                            </span>
                        );
                        })}
                        </div>
                     )}
                </div>
             </div>
        </div>
      </div>
    </div>
  );
}
