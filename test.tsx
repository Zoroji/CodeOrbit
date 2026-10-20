import React, { useState, useEffect, useCallback, useMemo } from "react";

// ─── Interfaces & Types ───────────────────────────────────────────────────────
export interface UserProfile {
  id: string;
  username: string;
  email: string;
  role: "admin" | "developer" | "viewer";
  avatarUrl?: string;
  isActive: boolean;
  metadata: {
    lastLogin: string;
    totalCommits: number;
    assignedRepositories: string[];
  };
}

export interface DashboardMetrics {
  totalScans: number;
  averageBlastRadius: number;
  affectedEndpoints: number;
  untestedImpacts: number;
}

export type ThemeMode = "dark" | "light" | "system";

// ─── Global Constants ─────────────────────────────────────────────────────────
export const API_BASE_URL = "https://api.codeorbit.dev/v1";
export const DEFAULT_PAGE_SIZE = 25;
export const MAX_RETRY_COUNT = 3;

// ─── Pure Helper Functions ────────────────────────────────────────────────────
export function formatRoleBadge(role: UserProfile["role"]): string {
  switch (role) {
    case "admin":
      return "badge-danger";
    case "developer":
      return "badge-primary";
    case "viewer":
    default:
      return "badge-secondary";
  }
}

export const calculateRiskLevel = (metrics: DashboardMetrics): "LOW" | "MEDIUM" | "HIGH" => {
  if (metrics.untestedImpacts > 5 || metrics.affectedEndpoints > 10) {
    return "HIGH";
  }
  if (metrics.untestedImpacts > 0 || metrics.affectedEndpoints > 3) {
    return "MEDIUM";
  }
  return "LOW";
};

// ─── Sub-Component: Status Notification Banner ───────────────────────────────
export function NotificationBanner({
  message,
  type,
  onDismiss,
}: {
  message: string;
  type: "success" | "warning" | "error";
  onDismiss: () => void;
}) {
  const alertClass =
    type === "error"
      ? "alert-danger"
      : type === "warning"
      ? "alert-warning"
      : "alert-success";

  return (
    <div className={`alert ${alertClass} d-flex justify-content-between align-items-center mb-3`} role="alert">
      <span>{message}</span>
      <button type="button" className="btn-close" aria-label="Close" onClick={onDismiss} />
    </div>
  );
}

// ─── Main Functional Component: UserDashboard ────────────────────────────────
export const UserDashboard: React.FC<{ initialRepoId?: string }> = ({ initialRepoId }) => {
  // State
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorNotification, setErrorNotification] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);
  const [theme, setTheme] = useState<ThemeMode>("dark");

  // Effect: Fetch Users on mount or repo change
  useEffect(() => {
    let isMounted = true;
    const fetchDashboardData = async () => {
      setIsLoading(true);
      setErrorNotification(null);
      try {
        const response = await fetch(`${API_BASE_URL}/users?repo=${initialRepoId || "all"}`);
        if (!response.ok) {
          throw new Error(`Failed to load users: HTTP ${response.status}`);
        }
        const data = await response.json();
        if (isMounted) {
          setUsers(data.users || []);
          setMetrics(data.metrics || null);
        }
      } catch (err: any) {
        if (isMounted) {
          setErrorNotification(err.message || "Failed to fetch dashboard data");
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchDashboardData();

    return () => {
      isMounted = false;
    };
  }, [initialRepoId]);

  // Handlers
  const handleUserSelect = useCallback((user: UserProfile) => {
    setSelectedUser(user);
  }, []);

  const handleToggleUserStatus = useCallback(async (userId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/users/${userId}/toggle-status`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Toggle status failed");
      setUsers((prev) =>
        prev.map((u) => (u.id === userId ? { ...u, isActive: !u.isActive } : u))
      );
    } catch (err: any) {
      setErrorNotification(err.message || "Status toggle failed");
    }
  }, []);

  // Filtered Users Memo
  const filteredUsers = useMemo(() => {
    if (!searchQuery.trim()) return users;
    const q = searchQuery.toLowerCase();
    return users.filter(
      (u) =>
        u.username.toLowerCase().includes(q) ||
        u.email.toLowerCase().includes(q) ||
        u.role.toLowerCase().includes(q)
    );
  }, [users, searchQuery]);

  // Render HTML / JSX
  return (
    <div className={`container-fluid p-4 ${theme === "dark" ? "bg-dark text-light" : "bg-light text-dark"}`}>
      {errorNotification && (
        <NotificationBanner
          message={errorNotification}
          type="error"
          onDismiss={() => setErrorNotification(null)}
        />
      )}

      {/* Header Bar */}
      <header className="d-flex justify-content-between align-items-center pb-3 mb-4 border-bottom">
        <div>
          <h1 className="h3 mb-0">CodeOrbit Developer Workbench</h1>
          <p className="text-muted small mb-0">Repository: {initialRepoId || "Global Space"}</p>
        </div>
        <div className="d-flex gap-2">
          <input
            type="text"
            className="form-control form-control-sm"
            placeholder="Search developers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button
            className="btn btn-outline-secondary btn-sm"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            Toggle Theme ({theme})
          </button>
        </div>
      </header>

      {/* Metrics Row */}
      {metrics && (
        <div className="row g-3 mb-4">
          <div className="col-md-3">
            <div className="card shadow-sm p-3 border-0 bg-secondary bg-opacity-10">
              <h6 className="text-muted small">Total AST Scans</h6>
              <h2 className="mb-0 fw-bold">{metrics.totalScans}</h2>
            </div>
          </div>
          <div className="col-md-3">
            <div className="card shadow-sm p-3 border-0 bg-secondary bg-opacity-10">
              <h6 className="text-muted small">Average Blast Radius</h6>
              <h2 className="mb-0 fw-bold">{metrics.averageBlastRadius} hops</h2>
            </div>
          </div>
          <div className="col-md-3">
            <div className="card shadow-sm p-3 border-0 bg-secondary bg-opacity-10">
              <h6 className="text-muted small">Affected Endpoints</h6>
              <h2 className="mb-0 fw-bold text-warning">{metrics.affectedEndpoints}</h2>
            </div>
          </div>
          <div className="col-md-3">
            <div className="card shadow-sm p-3 border-0 bg-secondary bg-opacity-10">
              <h6 className="text-muted small">Untested Impact Risk</h6>
              <span className={`badge ${calculateRiskLevel(metrics) === "HIGH" ? "bg-danger" : "bg-success"}`}>
                {calculateRiskLevel(metrics)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Main Table Content */}
      {isLoading ? (
        <div className="text-center py-5">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading repository members...</span>
          </div>
        </div>
      ) : (
        <div className="table-responsive shadow-sm rounded">
          <table className="table table-hover align-middle mb-0">
            <thead className="table-dark">
              <tr>
                <th>Developer</th>
                <th>Role</th>
                <th>Commits</th>
                <th>Last Active</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((user) => (
                <tr key={user.id} onClick={() => handleUserSelect(user)} style={{ cursor: "pointer" }}>
                  <td>
                    <strong>{user.username}</strong>
                    <div className="text-muted small">{user.email}</div>
                  </td>
                  <td>
                    <span className={`badge ${formatRoleBadge(user.role)}`}>{user.role}</span>
                  </td>
                  <td>{user.metadata.totalCommits}</td>
                  <td>{user.metadata.lastLogin}</td>
                  <td>
                    <span className={`badge ${user.isActive ? "bg-success" : "bg-secondary"}`}>
                      {user.isActive ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn btn-sm btn-outline-info"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleUserStatus(user.id);
                      }}
                    >
                      Toggle
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default UserDashboard;
