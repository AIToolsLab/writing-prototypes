import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { 
  Home, 
  Highlighter, 
  FileEdit, 
  Type, 
  Wrench, 
  Menu, 
  ChevronLeft
} from 'lucide-react';

export default function Layout() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { to: "/", icon: Home, label: "Home" },
    { to: "/rewrite", icon: FileEdit, label: "Rewrite" },
    { to: "/highlights", icon: Highlighter, label: "Highlights" },
    { to: "/type-assistant", icon: Type, label: "Type Assistant" },
    { to: "/internals", icon: Wrench, label: "Internals" },
  ];

  return (
    <div className="flex h-screen w-full bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <aside 
        className={`${
          isCollapsed ? 'w-20' : 'w-64'
        } bg-white border-r border-gray-200 transition-all duration-300 ease-in-out flex flex-col h-full shadow-sm z-10 relative`}
      >
        {/* Sidebar Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-gray-100">
          <div className={`font-bold text-xl text-sky-600 truncate ${isCollapsed ? 'hidden' : 'block'}`}>
            Writing Tools
          </div>
          <button 
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 rounded-md hover:bg-gray-100 text-gray-500 transition-colors"
          >
            {isCollapsed ? <Menu size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 overflow-y-auto py-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `
                flex items-center px-4 py-3 mx-2 rounded-md transition-colors group
                ${isActive 
                  ? 'bg-sky-50 text-sky-700' 
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'}
              `}
              title={isCollapsed ? item.label : undefined}
            >
              <item.icon size={20} className={`${isCollapsed ? 'mx-auto' : 'mr-3'} flex-shrink-0 transition-all`} />
              <span className={`font-medium whitespace-nowrap overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>
                {item.label}
              </span>
              
              {/* Tooltip for collapsed mode */}
              {isCollapsed && (
                 <div className="absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50 whitespace-nowrap">
                  {item.label}
                </div>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer / User section */}
        <div className="p-4 border-t border-gray-100">
           <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'justify-start space-x-3'}`}>
              <div className="w-8 h-8 rounded-full bg-sky-100 flex items-center justify-center text-sky-600 font-bold text-sm">
                AI
              </div>
              <div className={`overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>
                <p className="text-sm font-medium text-gray-900">User</p>
                <p className="text-xs text-gray-500">Writing Assistant</p>
              </div>
           </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Optional Top Bar */}
        {/* <header className="h-16 bg-white border-b border-gray-200 flex items-center px-8 justify-between shrink-0">
             <h1 className="text-lg font-semibold text-gray-800">
               {navItems.find(i => i.to === location.pathname)?.label || 'Dashboard'}
             </h1>
        </header> */}

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth">
           <div className="max-w-5xl mx-auto w-full animate-in fade-in zoom-in duration-300">
              <Outlet />
           </div>
        </div>
      </main>
    </div>
  );
}
