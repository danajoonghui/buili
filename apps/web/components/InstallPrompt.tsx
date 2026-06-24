"use client";

import { Download, Smartphone } from "lucide-react";
import { useEffect, useState } from "react";

type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
};

export function InstallPrompt({ onNotice }: { onNotice?: (message: string) => void }) {
  const [event, setEvent] = useState<BeforeInstallPromptEvent | null>(null);
  const [installed, setInstalled] = useState(false);
  const [helpVisible, setHelpVisible] = useState(false);

  useEffect(() => {
    const handler = (incoming: Event) => {
      incoming.preventDefault();
      setEvent(incoming as BeforeInstallPromptEvent);
    };
    const installedHandler = () => setInstalled(true);
    window.addEventListener("beforeinstallprompt", handler);
    window.addEventListener("appinstalled", installedHandler);
    return () => {
      window.removeEventListener("beforeinstallprompt", handler);
      window.removeEventListener("appinstalled", installedHandler);
    };
  }, []);

  if (installed) {
    return (
      <span className="install-state">
        <Smartphone size={16} />
        Installed
      </span>
    );
  }

  return (
    <button
      className="icon-button install-button"
      type="button"
      aria-label="Install Buili"
      title="Install Buili"
      onClick={async () => {
        if (!event) {
          setHelpVisible(true);
          onNotice?.("Install is available from the browser address bar or share menu on this device.");
          return;
        }
        await event.prompt();
        const choice = await event.userChoice;
        onNotice?.(
          choice.outcome === "accepted" ? "Buili install started." : "Install was dismissed by the browser."
        );
        setEvent(null);
      }}
    >
      <Download size={18} />
      <span>Install</span>
      {helpVisible ? <small className="install-help">Use the browser install icon or share menu.</small> : null}
    </button>
  );
}
