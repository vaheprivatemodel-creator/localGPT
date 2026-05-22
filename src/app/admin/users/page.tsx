"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { chatAPI } from "@/lib/api";

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  created_at: string;
  is_active: number;
}

const ROLE_LABELS: Record<string, string> = {
  admin: "Admin",
  attorney: "Attorney",
  paralegal: "Paralegal",
};

const ROLE_COLORS: Record<string, string> = {
  admin: "bg-purple-900/40 text-purple-300 border-purple-700/40",
  attorney: "bg-blue-900/40 text-blue-300 border-blue-700/40",
  paralegal: "bg-gray-800 text-gray-300 border-gray-700",
};

export default function AdminUsersPage() {
  const { user, isLoading } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<User[]>([]);
  const [fetching, setFetching] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [resetTarget, setResetTarget] = useState<User | null>(null);
  const [form, setForm] = useState({ email: "", name: "", role: "attorney", password: "" });
  const [resetPw, setResetPw] = useState("");
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!isLoading && user?.role !== "admin") router.replace("/");
  }, [isLoading, user, router]);

  const load = async () => {
    try {
      const data = await chatAPI.listUsers();
      setUsers(data.users);
    } catch (e) {
      console.error(e);
    } finally {
      setFetching(false);
    }
  };

  useEffect(() => { if (!isLoading) load(); }, [isLoading]);

  const flash = (text: string, ok: boolean) => {
    setMsg({ text, ok });
    setTimeout(() => setMsg(null), 4000);
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await chatAPI.createUser(form);
      flash("User created successfully", true);
      setShowCreate(false);
      setForm({ email: "", name: "", role: "attorney", password: "" });
      load();
    } catch (err: unknown) {
      flash(err instanceof Error ? err.message : "Failed to create user", false);
    }
  };

  const toggleActive = async (u: User) => {
    try {
      await chatAPI.updateUser(u.id, { is_active: !u.is_active });
      load();
    } catch (err: unknown) {
      flash(err instanceof Error ? err.message : "Failed to update user", false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resetTarget) return;
    try {
      await chatAPI.updateUser(resetTarget.id, { password: resetPw });
      flash("Password reset successfully", true);
      setResetTarget(null);
      setResetPw("");
    } catch (err: unknown) {
      flash(err instanceof Error ? err.message : "Failed to reset password", false);
    }
  };

  if (isLoading || user?.role !== "admin") return null;

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={() => router.push("/")} className="text-gray-400 hover:text-white text-sm transition">
            ← Back
          </button>
          <h1 className="text-lg font-semibold">User Management</h1>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition"
        >
          + Add User
        </button>
      </header>

      {/* Flash message */}
      {msg && (
        <div className={`mx-6 mt-4 rounded-lg px-4 py-3 text-sm border ${msg.ok ? "bg-green-900/40 text-green-300 border-green-700/40" : "bg-red-900/40 text-red-300 border-red-700/40"}`}>
          {msg.text}
        </div>
      )}

      {/* Table */}
      <div className="px-6 py-6">
        {fetching ? (
          <p className="text-gray-500 text-sm">Loading users…</p>
        ) : (
          <div className="rounded-xl border border-gray-800 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-900 text-gray-400 text-xs uppercase tracking-wider">
                <tr>
                  <th className="px-5 py-3 text-left">Name</th>
                  <th className="px-5 py-3 text-left">Email</th>
                  <th className="px-5 py-3 text-left">Role</th>
                  <th className="px-5 py-3 text-left">Status</th>
                  <th className="px-5 py-3 text-left">Created</th>
                  <th className="px-5 py-3 text-left">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {users.map((u) => (
                  <tr key={u.id} className="bg-black hover:bg-gray-900/50 transition">
                    <td className="px-5 py-4 font-medium">{u.name}</td>
                    <td className="px-5 py-4 text-gray-400">{u.email}</td>
                    <td className="px-5 py-4">
                      <span className={`inline-flex text-xs font-medium px-2 py-0.5 rounded-full border ${ROLE_COLORS[u.role] || ROLE_COLORS.paralegal}`}>
                        {ROLE_LABELS[u.role] || u.role}
                      </span>
                    </td>
                    <td className="px-5 py-4">
                      <span className={`inline-flex text-xs font-medium px-2 py-0.5 rounded-full ${u.is_active ? "text-green-400" : "text-gray-500"}`}>
                        {u.is_active ? "Active" : "Disabled"}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-gray-500">
                      {new Date(u.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-3">
                        <button
                          onClick={() => setResetTarget(u)}
                          className="text-gray-400 hover:text-white text-xs underline transition"
                        >
                          Reset password
                        </button>
                        <button
                          onClick={() => toggleActive(u)}
                          className={`text-xs transition ${u.is_active ? "text-red-400 hover:text-red-300" : "text-green-400 hover:text-green-300"}`}
                        >
                          {u.is_active ? "Disable" : "Enable"}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create user modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-7 w-full max-w-md shadow-2xl">
            <h2 className="text-base font-semibold mb-5">Create New User</h2>
            <form onSubmit={handleCreate} className="space-y-4">
              {[
                { label: "Full name", key: "name", type: "text", placeholder: "Jane Smith" },
                { label: "Email", key: "email", type: "email", placeholder: "jane@firm.com" },
                { label: "Password", key: "password", type: "password", placeholder: "••••••••" },
              ].map(({ label, key, type, placeholder }) => (
                <div key={key}>
                  <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">{label}</label>
                  <input
                    type={type}
                    required
                    placeholder={placeholder}
                    value={form[key as keyof typeof form]}
                    onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                    className="w-full rounded-lg bg-gray-800 border border-gray-700 text-white placeholder-gray-500 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                  />
                </div>
              ))}
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">Role</label>
                <select
                  value={form.role}
                  onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
                  className="w-full rounded-lg bg-gray-800 border border-gray-700 text-white px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                >
                  <option value="paralegal">Paralegal</option>
                  <option value="attorney">Attorney</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div className="flex gap-3 pt-2">
                <button type="button" onClick={() => setShowCreate(false)} className="flex-1 py-2.5 rounded-lg border border-gray-700 text-gray-300 hover:text-white text-sm transition">Cancel</button>
                <button type="submit" className="flex-1 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Reset password modal */}
      {resetTarget && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-7 w-full max-w-sm shadow-2xl">
            <h2 className="text-base font-semibold mb-1">Reset Password</h2>
            <p className="text-sm text-gray-400 mb-5">for <strong className="text-white">{resetTarget.name}</strong></p>
            <form onSubmit={handleResetPassword} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">New Password</label>
                <input
                  type="password"
                  required
                  minLength={8}
                  placeholder="Min. 8 characters"
                  value={resetPw}
                  onChange={(e) => setResetPw(e.target.value)}
                  className="w-full rounded-lg bg-gray-800 border border-gray-700 text-white placeholder-gray-500 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                />
              </div>
              <div className="flex gap-3">
                <button type="button" onClick={() => { setResetTarget(null); setResetPw(""); }} className="flex-1 py-2.5 rounded-lg border border-gray-700 text-gray-300 hover:text-white text-sm transition">Cancel</button>
                <button type="submit" className="flex-1 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition">Reset</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
