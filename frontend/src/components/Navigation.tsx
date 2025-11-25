import { Link, useLocation } from "react-router-dom";
import { FileText, BarChart3 } from "lucide-react";

export default function Navigation() {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Extractor", icon: FileText },
    { path: "/graph", label: "Graph", icon: BarChart3 },
  ];

  return (
    <nav className="main-navigation">
      <div className="nav-brand">
        <div className="logo" />
        <div className="title">CloudExtel Extractor</div>
      </div>
      <div className="nav-links">
        {navItems.map(({ path, label, icon: Icon }) => {
          const isActive = location.pathname === path;
          return (
            <Link
              key={path}
              to={path}
              className={`nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={20} />
              <span>{label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

