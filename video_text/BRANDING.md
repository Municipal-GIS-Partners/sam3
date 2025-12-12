# MGP Brand Implementation

This document outlines the Municipal GIS Partners (MGP) brand styling applied to the SAM3 Video Text-Prompt Segmentation application.

## Brand Colors Applied

### Primary Colors
- **MGP Blue (#005f9d)**: Primary buttons, headings, navigation elements, slider controls
- **MGP Orange (#ee5623)**: Accent color for borders, active states, highlights

### Secondary Colors
- **Charcoal (#46525E)**: Body text, labels
- **Slate (#5F6A72)**: Secondary text, hints
- **Ocean Blue (#006699)**: Hover states

### Accent Colors
- **Gold (#F2B225)**: Secondary buttons (download actions)
- **Light Background (#F5F5F5)**: Page background
- **White (#FFFFFF)**: Card backgrounds
- **Border Gray (#E0E0E0)**: Borders and dividers

## Typography

### Font Family
**Nunito Sans** - Loaded from Google Fonts with weights:
- 300 (Light) - Large headings (h1)
- 400 (Regular) - Body text
- 600 (Semi-bold) - Buttons, labels, subheadings
- 800 (Extra-bold) - Emphasis elements

### Heading Hierarchy
```
h1: 48px, weight 300 (Light)
h2: 30px, weight 300 (Light)
h3: 20px, weight 600 (Semi-bold)
Body: 16px, weight 400 (Regular)
```

## Component Styling

### Header
- MGP Blue background (#005f9d)
- White text on blue
- Clean, professional appearance

### Section Headers
- MGP Blue text color
- MGP Orange bottom border (3px)
- Creates visual hierarchy

### Buttons

**Primary Buttons**
- Background: MGP Blue
- Text: White
- Shadow: Subtle rgba shadow
- Hover: Darker blue with lift effect

**Secondary Buttons**
- Background: MGP Gold
- Text: Charcoal
- Used for download actions

**Navigation Buttons**
- Background: MGP Blue
- Compact sizing
- Ocean blue hover state

### Form Controls

**Text Inputs**
- Border: Gray with blue focus state
- Focus ring: Subtle blue glow
- Rounded corners (4px)

**Sliders**
- Track: Border gray
- Thumb: MGP Blue circle with shadow
- Hover: Ocean blue with scale effect
- Value display: MGP Blue, bold

### Progress Bar
- Container: Light background with border
- Fill: MGP Blue
- Animated shimmer effect
- Clean, modern appearance

### Cards & Panels
- White background
- Subtle shadows
- Border: Light gray
- Blue left border on info items

### Thumbnails
- Gray border default
- Blue border on hover
- Orange border when active
- Smooth transitions

## Visual Identity Elements

### Color Usage Strategy
1. **Blue Dominance**: Primary brand color for key actions and navigation
2. **Orange Accents**: Strategic use for highlighting and drawing attention
3. **Neutral Base**: Clean gray/white foundation for readability
4. **Gold for Secondary**: Distinguishes download actions from primary actions

### Design Principles Applied
1. **Clean & Professional**: Ample white space, clear hierarchy
2. **Accessible**: High contrast ratios for readability
3. **Consistent**: Uniform application of brand colors
4. **Modern**: Subtle shadows, rounded corners, smooth transitions
5. **Data-Focused**: Clear visual separation for comparing images

## Accessibility Features

- WCAG AA compliant contrast ratios
- Focus states on all interactive elements
- Semantic HTML structure
- ARIA labels for hidden elements
- Keyboard navigation support

## Animation & Transitions

- 0.3s ease transitions on interactive elements
- Subtle hover lifts on buttons and thumbnails
- Shimmer effect on progress bar
- Smooth scrolling to sections

## Responsive Behavior

- Flexible grid layouts
- Mobile-first approach
- Breakpoints at 968px and 640px
- Maintains brand consistency across devices

## File Updates

### CSS (`static/css/style.css`)
- Added CSS variables for brand colors
- Updated all color references
- Changed font family to Nunito Sans
- Revised component styles

### HTML (`templates/index.html`)
- Added Google Fonts link
- Updated page title
- Replaced inline styles with classes
- Added accessibility attributes

### JavaScript (`static/js/app.js`)
- Updated to use `.hidden` class
- Removed style.display manipulation
- Consistent with CSS approach

## Brand Compliance

This implementation follows the MGP brand guidelines:
- ✅ Correct color palette usage
- ✅ Nunito Sans typography
- ✅ Professional appearance
- ✅ Consistent application
- ✅ Accessible design
- ✅ Modern UI patterns

## Maintenance Notes

To maintain brand consistency:
1. Always use CSS variables for colors
2. Follow the established heading hierarchy
3. Use Nunito Sans font weights appropriately
4. Apply the `.btn-primary` and `.btn-secondary` classes for buttons
5. Maintain the blue/orange accent pattern
6. Keep border radius at 4px for consistency
7. Use established shadow patterns

---

**Brand Style Guide Reference**: Based on Municipal GIS Partners color and font standards
**Last Updated**: December 2024
