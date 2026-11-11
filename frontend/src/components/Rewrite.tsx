
import { useState, useEffect, useCallback } from 'react';
import { getPredsApi } from '../api';
import { RefreshCw, Play } from 'lucide-react';

const PROMPT_OPTIONS = [
  "Rewrite this document to be more clear and concise.",
  "Rewrite this document to be more detailed and engaging.",
  "Summarize this document in one sentence.",
  "Translate this document into French.",
  "Translate this document into Spanish.",
];

export default function Rewrite() {
  const [prompt, setPrompt] = useState(PROMPT_OPTIONS[0]);
  const [customPrompt, setCustomPrompt] = useState("");
  const [doc, setDoc] = useState("");
  const [rewriteInProgress, setRewriteInProgress] = useState("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [predictions, setPredictions] = useState<[string, number][]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPredictions = useCallback(async () => {
    if (!doc.trim()) return;

    setIsLoading(true);
    try {
      // If custom prompt is used, send that
      const p = prompt === "Other" ? customPrompt : prompt;
      const preds = await getPredsApi(p, doc, rewriteInProgress);
      setPredictions(preds || []); // Ensure array
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, [prompt, customPrompt, doc, rewriteInProgress]);

  // Debouce effect
  useEffect(() => {
    const timeout = setTimeout(fetchPredictions, 800);
    return () => clearTimeout(timeout);
  }, [fetchPredictions]);

  const handleAppendToken = (token: string) => {
    setRewriteInProgress(prev => prev + token);
  };

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-gray-900">Rewrite with Predictive Text</h2>
        <div className="flex space-x-2">
            <button 
                onClick={fetchPredictions}
                disabled={isLoading}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
                <RefreshCw size={16} className={`mr-1.5 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
            </button>
        </div>
      </div>

      {/* Controls Card */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Task / Prompt</label>
                <select 
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={prompt} 
                onChange={e => setPrompt(e.target.value)}
                >
                {PROMPT_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                <option value="Other">Other (Custom)</option>
                </select>
            </div>
            {prompt === "Other" && (
                 <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Custom Instruction</label>
                    <input 
                        type="text"
                        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="e.g. Turn this into a haiku..."
                        value={customPrompt}
                        onChange={e => setCustomPrompt(e.target.value)}
                    />
                 </div>
            )}
        </div>
      </div>

      {/* Editor Area */}
      <div className="grid gap-6 lg:grid-cols-2 h-[600px]">
        <div className="flex flex-col space-y-2 h-full">
          <label className="block text-sm font-medium text-gray-700">Original Document</label>
          <textarea 
            className="flex-1 w-full rounded-lg border border-gray-300 p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none shadow-sm"
            placeholder="Paste your document here..."
            value={doc} 
            onChange={e => setDoc(e.target.value)}
          />
        </div>
        
        <div className="flex flex-col space-y-2 h-full">
          <label className="block text-sm font-medium text-gray-700">Rewrite with Assistance</label>
          <div className="flex-1 flex flex-col rounded-lg border border-gray-300 bg-white shadow-sm overflow-hidden">
             <textarea 
                className="flex-1 w-full p-4 font-mono text-sm focus:outline-none border-t-0 focus:ring-0 resize-none border-b border-gray-100"
                placeholder="Start typing or click suggestions..." // Note: Predictions append naturally
                value={rewriteInProgress} 
                onChange={e => setRewriteInProgress(e.target.value)}
            />
            
            {/* Predictions Bar */}
            <div className="bg-gray-50 p-3 border-t border-gray-200">
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 flex items-center">
                    <Play size={12} className="mr-1" /> Next Token Predictions
                </div>
                <div className="flex flex-wrap gap-2">
                    {predictions.length === 0 && !isLoading && (
                        <span className="text-sm text-gray-400 italic">Start typing to see predictions...</span>
                    )}
                     {predictions.map((p, idx) => (
                        <button 
                            key={idx} 
                            onClick={() => handleAppendToken(p[0])}
                            className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-white border border-gray-200 text-gray-700 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-200 transition-colors shadow-sm"
                            title={`Logprob: ${p[1]}`}
                        >
                        {p[0] === '\n' ? '↵' : p[0]}
                        </button>
                    ))}
                </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
