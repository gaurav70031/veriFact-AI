/**
 * VeriFact AI brand logo icon — SVG inline component.
 * V shape + document card + magnifying glass with checkmark.
 * Use `size` to control width/height (square).
 */

interface LogoIconProps {
  size?: number
  className?: string
}

export function LogoIcon({ size = 36, className }: LogoIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="lv" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stopColor="#3B82F6" />
          <stop offset="100%" stopColor="#34D399" />
        </linearGradient>
        <linearGradient id="ld" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stopColor="#EFF6FF" />
          <stop offset="100%" stopColor="#BFDBFE" />
        </linearGradient>
      </defs>

      {/* ── V stroke ── */}
      <path
        d="M30 30 L70 130 L100 75 L130 130 L170 30"
        stroke="url(#lv)"
        strokeWidth="22"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* ── Document card ── */}
      <rect x="72" y="18" width="58" height="72" rx="6"
        fill="url(#ld)" stroke="#93C5FD" strokeWidth="1.5" opacity={0.92} />

      {/* Blue square thumbnail */}
      <rect x="79" y="27" width="14" height="14" rx="2" fill="#3B82F6" />

      {/* Text lines */}
      <rect x="98" y="29" width="24" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.7} />
      <rect x="98" y="36" width="20" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.5} />
      <rect x="79" y="47" width="43" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.6} />
      <rect x="79" y="55" width="38" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.5} />
      <rect x="79" y="63" width="43" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.4} />
      <rect x="79" y="71" width="30" height="3.5" rx="1.5" fill="#1E3A5F" opacity={0.3} />

      {/* ── Magnifying glass ring ── */}
      <circle cx="148" cy="72" r="22" fill="#0D1B2A" stroke="#34D399" strokeWidth="2.5" />
      <circle cx="148" cy="72" r="20" fill="none" stroke="#34D399" strokeWidth="3.5" />

      {/* Checkmark inside magnifier */}
      <path
        d="M138 72 L145 79 L160 62"
        stroke="white"
        strokeWidth="4.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Handle */}
      <line x1="163" y1="87" x2="175" y2="100"
        stroke="#34D399" strokeWidth="6" strokeLinecap="round" />
    </svg>
  )
}
