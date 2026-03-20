
import { useState, useEffect, useRef } from 'react';
import { continueMessages } from '../api';
import { Send, User, Bot, RotateCcw, Play } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function TypeAssistant() {
  const [messages, setMessages] = useState<Message[]>([{ role: 'user', content: '' }]);
  const [continuations, setContinuations] = useState<string[]>([]); // Simplified to string array for now if api returns objects map them
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const lastMessageIndex = messages.length - 1;
  const activeMessage = messages[lastMessageIndex];

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, activeMessage.content]); // Scroll on new message or typing

  const handleUpdateActiveMessage = (text: string) => {
    setMessages(prev => {
      const newMessages = [...prev];
      newMessages[lastMessageIndex] = {
        ...newMessages[lastMessageIndex],
        content: text
      };
      return newMessages;
    });
  };

  const handleRewind = (index: number) => {
    if (confirm("Rewind to this point? Subsequent messages will be lost.")) {
         setMessages(prev => prev.slice(0, index + 1));
    }
  };

  const handleSend = () => {
    if (!activeMessage.content.trim()) return;
    const nextRole = activeMessage.role === 'user' ? 'assistant' : 'user';
    setMessages(prev => [...prev, { role: nextRole, content: '' }]);
    setContinuations([]);
  };

  // Mock API call for now since we don't have the real 'continueMessages' type perfectly defined 
  // or it might fail if backend isn't ready. usage looks like 'continueMessages(messages)'
  useEffect(() => {
    let active = true;
    const fetchContinuations = async () => {
      // Don't fetch if empty content or just sent
      if (!activeMessage.content && messages.length > 1) return; 

      setIsLoading(true);
      try {
        // Assume API returns { doc_text: string }[] or similar
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result: any = await continueMessages(messages);
        if (active && result) {
            // Adapt based on actual API return shape (guessing array of objects with doc_text)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const texts = Array.isArray(result) ? result.map((r: any) => r.doc_text || r) : [];
            setContinuations(texts);
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (active) setIsLoading(false);
      }
    };

    const timeout = setTimeout(fetchContinuations, 600);
    return () => {
      clearTimeout(timeout);
      active = false;
    };
  }, [messages, activeMessage.content]); 

  const appendToken = (token: string) => {
    handleUpdateActiveMessage(activeMessage.content + token);
  };

  return (
    <div className="max-w-4xl mx-auto p-4 md:p-6 h-[calc(100vh-100px)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold tracking-tight text-gray-900">Interactive Assistant</h2>
        <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded border border-gray-200">
            {messages.length} turns
        </span>
      </div>
      
      {/* Chat History Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-4 p-4 rounded-xl border border-gray-200 bg-gray-50 mb-4 shadow-inner"
      >
        {messages.slice(0, -1).map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex max-w-[80%] ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center mx-2 ${m.role === 'user' ? 'bg-blue-100 text-blue-600' : 'bg-green-100 text-green-600'}`}>
                    {m.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div 
                    className={`relative p-3 rounded-lg text-sm shadow-sm border ${
                        m.role === 'user' 
                        ? 'bg-blue-600 text-white border-blue-600 rounded-tr-none' 
                        : 'bg-white text-gray-800 border-gray-200 rounded-tl-none'
                    }`}
                >
                    <p className="whitespace-pre-wrap">{m.content}</p>
                    <button 
                        onClick={() => handleRewind(i)}
                        className={`absolute -bottom-5 ${m.role === 'user' ? 'right-0' : 'left-0'} text-xs text-gray-400 hover:text-gray-600 flex items-center opacity-0 hover:opacity-100 transition-opacity`}
                        title="Rewind to here"
                    >
                        <RotateCcw size={10} className="mr-1" /> Edit
                    </button>
                </div>
            </div>
          </div>
        ))}
        {/* Placeholder for active message being typed if it's not the first one */}
        {messages.length > 0 && (
             <div className={`flex ${activeMessage.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex w-full max-w-[90%] ${activeMessage.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center mx-2 ${activeMessage.role === 'user' ? 'bg-blue-100 text-blue-600' : 'bg-green-100 text-green-600'}`}>
                        {activeMessage.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div className="flex-1">
                        <div className="relative rounded-lg border border-gray-300 bg-white shadow-sm overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent">
                            <textarea
                                className="w-full p-3 min-h-[100px] text-sm focus:outline-none resize-none"
                                value={activeMessage.content}
                                onChange={e => handleUpdateActiveMessage(e.target.value)}
                                placeholder={activeMessage.role === 'user' ? "Type your message..." : "Assistant is typing..."}
                                autoFocus
                            />
                            {/* Suggestions Bar */}
                            {(continuations.length > 0 || isLoading) && (
                                <div className="bg-gray-50 px-3 py-2 border-t border-gray-100 flex items-center gap-2 overflow-x-auto">
                                    <Play size={12} className="text-gray-400 flex-shrink-0" />
                                    {isLoading && continuations.length === 0 && <span className="text-xs text-gray-400 animate-pulse">Predicting...</span>}
                                    {continuations.map((text, idx) => (
                                        <button 
                                            key={idx}
                                            onClick={() => appendToken(text)}
                                            className="whitespace-nowrap px-2 py-1 rounded bg-white border border-gray-200 text-xs text-gray-700 hover:bg-blue-50 hover:text-blue-600 hover:border-blue-200 transition-colors shadow-sm"
                                        >
                                            {text}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                         <div className="mt-2 flex justify-end">
                            <button 
                                onClick={handleSend}
                                disabled={!activeMessage.content.trim()}
                                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
                            >
                                <Send size={16} className="mr-2" />
                                {activeMessage.role === 'user' ? 'Send' : 'Continue'}
                            </button>
                        </div>
                    </div>
                </div>
             </div>
        )}
      </div>
    </div>
  );
}
