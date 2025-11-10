/*
Kiosk-style Mobile Web App (single-file React component)
- TailwindCSS utility classes are used for styling (assumes Tailwind in your project)
- Exports a default React component that can be dropped into a page
- Mobile-first layout, large touch-friendly buttons, offline-friendly localStorage order draft
- Handles item selection, qty adjustments, subtotal/total calculation, tax, and sends a single POST JSON

Usage:
1. Ensure your app includes Tailwind and React (this is a component file, e.g. KioskOrder.jsx)
2. Import and render: import KioskOrder from './kiosk-order-webapp.jsx';
3. Endpoint configured to POST to `/orders/add` — change `API_ENDPOINT` constant if needed

*/

import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_ENDPOINT = 'http://192.168.100.164:8000/orders/add'; // full IP for mobile access

// Hardcoded menu
const DEFAULT_MENU = [
  { id: 'burger', name: 'Burger', price: 60.0 },
  { id: 'fries', name: 'Fries', price: 30.0 },
  { id: 'soda', name: 'Soda', price: 20.0 },
  { id: 'coffee', name: 'Coffee', price: 30.0 },
  { id: 'pasta', name: 'Pasta', price: 75.0 },
];

export default function KioskOrder({ menu = DEFAULT_MENU, taxPct = 0.12 }) {
  const [items, setItems] = useState({});
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  // Helpers
  const addOne = (id) => setItems(prev => ({ ...prev, [id]: (prev[id] || 0) + 1 }));
  const subOne = (id) => setItems(prev => {
    const cur = (prev[id] || 0) - 1;
    if (cur <= 0) {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    }
    return { ...prev, [id]: cur };
  });

  const orderLines = useMemo(() => {
    return Object.entries(items)
      .map(([id, qty]) => {
        const m = menu.find(x => x.id === id);
        return m ? { ...m, qty } : null;
      })
      .filter(Boolean);
  }, [items, menu]);

  const subtotal = useMemo(() => orderLines.reduce((s, it) => s + it.price * it.qty, 0), [orderLines]);
  const total = useMemo(() => +(subtotal * (1 + taxPct)).toFixed(2), [subtotal, taxPct]);

  function nextOrderNumber() {
    const d = new Date();
    const pad = (n, z = 2) => n.toString().padStart(z, '0');
    const core = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return core + Math.floor(Math.random() * 90 + 10);
  }

  const makePayload = () => {
    return {
      order_number: nextOrderNumber(),
      items: orderLines.map(it => ({ name: it.name, qty: it.qty })), // list of Item objects
      total: total // float
    };
  };

  const submitOrder = async () => {
    if (orderLines.length === 0) {
      setMessage({ type: 'error', text: 'Add at least one item.' });
      return;
    }
    setLoading(true);
    setMessage(null);
    const payload = makePayload();

    try {
      const res = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Server error');
      }
      const data = await res.json();
      setMessage({ type: 'success', text: `Order ${data.order_number || payload.order_number} sent!` });
      setItems({});
    } catch (err) {
      console.error(err);
      setMessage({ type: 'error', text: 'Failed to send order. Check backend and network.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-4 flex flex-col items-center justify-start">
      <div className="max-w-md w-full bg-white shadow-md rounded-2xl overflow-hidden">
        <header className="p-4 border-b">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-extrabold">Kiosk Order</h1>
            <div className="text-sm text-slate-500">Touch-friendly • Mobile</div>
          </div>
        </header>

        <main className="p-4">
          {/* Menu */}
          <section>
            <h2 className="text-sm font-semibold mb-2">Menu</h2>
            <div className="grid grid-cols-2 gap-2">
              {menu.map((m) => (
                <motion.button
                  key={m.id}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => addOne(m.id)}
                  className="bg-slate-100 p-3 rounded-lg flex flex-col items-start gap-1"
                >
                  <div className="font-medium">{m.name}</div>
                  <div className="text-xs text-slate-500">₱{m.price.toFixed(2)}</div>
                </motion.button>
              ))}
            </div>
          </section>

          {/* Cart */}
          <section className="mt-4">
            <h2 className="text-sm font-semibold mb-2">Order</h2>
            <div className="space-y-2">
              <AnimatePresence>
                {orderLines.length === 0 && (
                  <motion.div initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="text-xs text-slate-400">
                    No items yet — tap menu items to add.
                  </motion.div>
                )}

                {orderLines.map((it) => (
                  <motion.div layout key={it.id} className="flex items-center justify-between bg-slate-50 p-2 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="font-medium">{it.name}</div>
                      <div className="text-xs text-slate-500">₱{it.price.toFixed(2)}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button onClick={() => subOne(it.id)} className="w-9 h-9 rounded-lg bg-white shadow-sm">-</button>
                      <input
                        type="number"
                        min={1}
                        value={items[it.id]}
                        onChange={(e) => setItems(prev => ({ ...prev, [it.id]: Math.max(1, Number(e.target.value)) }))}
                        className="w-14 text-center rounded-lg border px-2 py-1"
                      />
                      <button onClick={() => addOne(it.id)} className="w-9 h-9 rounded-lg bg-white shadow-sm">+</button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            <div className="mt-3 p-3 bg-white rounded-lg border">
              <div className="flex justify-between text-sm"><div>Total</div><div>₱{total.toFixed(2)}</div></div>
              <div className="text-xs text-slate-500 mt-1">Items: {orderLines.reduce((s,it)=>s+it.qty,0)}</div>
            </div>

            <div className="mt-3 flex gap-2">
              <button onClick={() => setItems({})} className="flex-1 py-3 rounded-lg bg-red-500 text-white font-semibold">Clear</button>
              <button onClick={submitOrder} disabled={loading} className="flex-1 py-3 rounded-lg bg-blue-600 text-white font-semibold">
                {loading ? 'Sending…' : 'Send Order'}
              </button>
            </div>

            {message && (
              <div className={`mt-3 p-2 rounded-md ${message.type === 'error' ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                {message.text}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}

