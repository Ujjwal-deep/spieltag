/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0a0a0b',
          800: '#141416',
          700: '#1c1c1f',
          600: '#27272a'
        },
        neon: {
          blue: '#3b82f6',
          green: '#10b981',
          red: '#ef4444',
          cyan: '#06b6d4'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
