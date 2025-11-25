import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import Home from "./pages/Home";
import Graph from "./pages/Graph";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        <header className="header">
          <Navigation />
          <div className="tagline">Marathi OCR with English translation</div>
        </header>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/graph" element={<Graph />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
