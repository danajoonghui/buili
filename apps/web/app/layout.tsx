import type { Metadata, Viewport } from "next";
import "./globals.css";
import { PwaRegistration } from "@/components/PwaRegistration";

export const metadata: Metadata = {
  title: "Buili",
  description: "Evidence-first construction issue review",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: "/buili_favicon_transparent.png",
    apple: "/buili_favicon_transparent.png"
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Buili"
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#f5f5f3"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>
        {children}
        <PwaRegistration />
      </body>
    </html>
  );
}
