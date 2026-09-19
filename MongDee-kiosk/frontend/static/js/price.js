// Shared price display: "ราคา  1,290  บาท" with the number set apart from the
// label/unit (and coloured red in CSS) instead of a "฿" glued to the digits.
function formatPriceNumber(n) {
  return new Intl.NumberFormat("th-TH", { minimumFractionDigits: 0, maximumFractionDigits: 2 }).format(n);
}

function buildPrice(price) {
  const wrap = document.createElement("span");
  wrap.className = "price-line";
  for (const [className, text] of [["price-label", "ราคา"], ["price-num", formatPriceNumber(price)], ["price-unit", "บาท"]]) {
    const part = document.createElement("span");
    part.className = className;
    part.textContent = text;
    wrap.append(part);
  }
  return wrap;
}
