import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import SpeedGame from "@/components/SpeedGame";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "41 Speed — Hand Crossing Rep Counter" },
      { name: "description", content: "Webcam game: count how many fast hand swaps you can do in 20 seconds. Real-time tracking, no install." },
    ],
  }),
});

function Index() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-background text-foreground">
        <div className="text-sm text-white/60">Loading…</div>
      </div>
    );
  }
  return <SpeedGame />;
}
