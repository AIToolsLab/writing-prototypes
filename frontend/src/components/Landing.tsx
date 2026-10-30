
import { Link } from 'react-router-dom';
import { PenTool, Highlighter, Type, Wrench, ArrowRight } from 'lucide-react';

export default function Landing() {
  return (
    <div className="max-w-4xl mx-auto space-y-12 py-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl">
          Writing Tools Prototypes
        </h1>
        <p className="text-xl text-gray-500 max-w-2xl mx-auto">
          Explore AI-powered writing assistance tools developed by{' '}
          <a href="https://kenarnold.org" className="text-blue-600 hover:text-blue-700 font-medium">Ken Arnold</a> and the{' '}
          <a href="https://thoughtful-ai.com/" className="text-blue-600 hover:text-blue-700 font-medium">
            Thoughtful AI Tools Lab
          </a> at Calvin University.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Link to="/rewrite" className="group block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-blue-300 transition-all duration-200">
          <div className="flex items-start justify-between">
            <div className="h-12 w-12 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <PenTool size={24} />
            </div>
            <ArrowRight size={20} className="text-gray-300 group-hover:text-blue-500 transition-colors" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors">Rewrite with Predictions</h3>
          <p className="text-gray-500 text-sm">Get granular control over rewriting text seeded by predictive tokens.</p>
        </Link>

        <Link to="/highlights" className="group block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-green-300 transition-all duration-200">
          <div className="flex items-start justify-between">
            <div className="h-12 w-12 bg-green-100 text-green-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Highlighter size={24} />
            </div>
            <ArrowRight size={20} className="text-gray-300 group-hover:text-green-500 transition-colors" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-green-600 transition-colors">Highlighter</h3>
          <p className="text-gray-500 text-sm">Visualize model uncertainty to identify areas that might need editing.</p>
        </Link>

        <Link to="/type-assistant" className="group block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-purple-300 transition-all duration-200">
          <div className="flex items-start justify-between">
            <div className="h-12 w-12 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Type size={24} />
            </div>
            <ArrowRight size={20} className="text-gray-300 group-hover:text-purple-500 transition-colors" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-purple-600 transition-colors">Type Assistant</h3>
          <p className="text-gray-500 text-sm">Interactive chat and typing assistance with next-token prediction.</p>
        </Link>

        <Link to="/internals" className="group block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-gray-300 transition-all duration-200">
          <div className="flex items-start justify-between">
            <div className="h-12 w-12 bg-gray-100 text-gray-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Wrench size={24} />
            </div>
            <ArrowRight size={20} className="text-gray-300 group-hover:text-gray-500 transition-colors" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-gray-600 transition-colors">Internals</h3>
          <p className="text-gray-500 text-sm">Debug view to inspect log probabilities and model internals.</p>
        </Link>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 text-sm text-blue-800 flex items-start">
        <span className="mr-2 text-xl">ℹ️</span>
        <p>
          *Note*: These services send data to a remote server for processing. The
          server logs requests. Don't use sensitive or identifiable information on
          this page.
        </p>
      </div>
    </div>
  );
}
