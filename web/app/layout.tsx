import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aria - Voice AI Support Agent",
  description: "Real-time voice qualification and support agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
