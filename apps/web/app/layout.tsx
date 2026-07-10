import type { Metadata, Viewport } from "next";
import "./globals.css";
import { PwaRegistration } from "@/components/PwaRegistration";

export const metadata: Metadata = {
  title: "BUILI — Construction Verification Intelligence",
  description: "Turn field observations into review-ready, source-cited construction issue packages.",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: "/brand/buili-mark.png",
    apple: "/brand/buili-mark.png"
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
    <html lang="en">
      <body>
        {children}
        <PwaRegistration />
      </body>
    </html>
  );
}
