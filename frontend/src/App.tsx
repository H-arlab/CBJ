import { useEffect } from 'react';
import { usePage } from './store/page';
import TopNav from './components/TopNav';
import Sidebar from './components/Sidebar';
import PublicationBar from './components/PublicationBar';
import Canvas from './components/Canvas';
import Library from './components/Library';
import FocusOverlay from './components/FocusOverlay';
import Drawer from './components/Drawer';
import Toast from './components/Toast';
import GlobalDropZone from './components/GlobalDropZone';
import ErrorBoundary from './components/ErrorBoundary';

export default function App() {
  const closeDrawer = usePage((s) => s.closeDrawer);
  const closeMapper = usePage((s) => s.closeMapper);
  const focusCell = usePage((s) => s.focusCell);

  // Always use publication styling; body class kept for CSS hooks.
  useEffect(() => {
    document.body.classList.add('pub');
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        closeDrawer();
        closeMapper();
        focusCell(null);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [closeDrawer, closeMapper, focusCell]);

  return (
    <ErrorBoundary>
      <div className="app">
        <TopNav />
        <Sidebar />
        <div className="workarea">
          <PublicationBar />
          <Canvas />
        </div>
        <aside className="chat-rail">
          <Library />
        </aside>
        <div className="footer">
          <span>H-WALKER CORE · v3</span>
          <span>PRETENDARD · JETBRAINS MONO</span>
        </div>
        <FocusOverlay />
        <Drawer />
        <Toast />
        <GlobalDropZone />
      </div>
    </ErrorBoundary>
  );
}
