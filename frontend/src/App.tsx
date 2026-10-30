
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Landing from './components/Landing';
import Rewrite from './components/Rewrite';
import Highlights from './components/Highlights';
import TypeAssistant from './components/TypeAssistant';
import ShowInternals from './components/ShowInternals';

function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Landing />} />
          <Route path="/rewrite" element={<Rewrite />} />
          <Route path="/highlights" element={<Highlights />} />
          <Route path="/type-assistant" element={<TypeAssistant />} />
          <Route path="/internals" element={<ShowInternals />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
