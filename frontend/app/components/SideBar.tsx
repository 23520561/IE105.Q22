const SideBar = function () {
  return (
    <aside className="bg-surface-container flex flex-col h-screen border-r border-white/5 docked left-0  w-64 sticky top-14 shrink-0">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-lg bg-surface-container-high flex items-center justify-center">
            <span
              className="material-symbols-outlined text-primary"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              account_tree
            </span>
          </div>
          <div>
            <h2 className="text-white font-headline font-bold text-sm tracking-tight">
              ML Pipeline v2
            </h2>
            <p className="text-[10px] uppercase tracking-widest text-on-surface-variant font-medium">
              Active Workspace
            </p>
          </div>
        </div>
        <nav className="space-y-1">
          <div className="flex items-center gap-3 px-3 py-2.5 font-['Inter'] text-xs font-medium bg-sky-500/10 text-sky-400 border-r-2 border-sky-400 cursor-pointer">
            <span className="material-symbols-outlined text-lg">
              folder_open
            </span>
            <span>Projects</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2.5 font-['Inter'] text-xs font-medium text-slate-400 hover:bg-surface-container-high hover:text-slate-200 transition-colors cursor-pointer">
            <span className="material-symbols-outlined text-lg">
              account_tree
            </span>
            <span>Nodes</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2.5 font-['Inter'] text-xs font-medium text-slate-400 hover:bg-surface-container-high hover:text-slate-200 transition-colors cursor-pointer">
            <span className="material-symbols-outlined text-lg">
              inventory_2
            </span>
            <span>Library</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2.5 font-['Inter'] text-xs font-medium text-slate-400 hover:bg-surface-container-high hover:text-slate-200 transition-colors cursor-pointer">
            <span className="material-symbols-outlined text-lg">history</span>
            <span>History</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2.5 font-['Inter'] text-xs font-medium text-slate-400 hover:bg-surface-container-high hover:text-slate-200 transition-colors cursor-pointer">
            <span className="material-symbols-outlined text-lg">inventory</span>
            <span>Archive</span>
          </div>
        </nav>
        <div className="mt-8">
          <button className="w-full py-2.5 primary-gradient text-on-primary-fixed font-bold text-xs rounded-lg active:scale-95 transition-all shadow-lg shadow-primary/10">
            Run Pipeline
          </button>
        </div>
      </div>
      <div className="mt-auto p-6 border-t border-white/5 space-y-1">
        <div className="flex items-center gap-3 px-3 py-2 font-['Inter'] text-xs font-medium text-slate-400 hover:text-white cursor-pointer">
          <span className="material-symbols-outlined text-lg">help</span>
          <span>Documentation</span>
        </div>
        <div className="flex items-center gap-3 px-3 py-2 font-['Inter'] text-xs font-medium text-slate-400 hover:text-white cursor-pointer">
          <span className="material-symbols-outlined text-lg">
            contact_support
          </span>
          <span>Support</span>
        </div>
      </div>
    </aside>
  );
};
export default SideBar;
