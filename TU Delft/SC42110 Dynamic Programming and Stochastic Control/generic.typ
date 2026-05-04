#import "@preview/ilm:1.4.1": *

#let emphasis_equbox(x) = [
    #box(
        inset: 8pt,
        fill: rgb("#f3f3f3")
    )[#x]
]

#let Cgr(x) = [ #text(fill: green, x) ]
#let Cre(x) = [ #text(fill: red, x) ]
#let Cbl(x) = [ #text(fill: blue, x) ]
#let Cpu(x) = [ #text(fill: purple, x) ]
#let Cgy(x) = [ #text(fill: gray, x) ]

#let resize_box(body) = layout(
    container => {
        let size = measure(body)
        let ratio = calc.min(container.width / size.width, container.height / size.height) * 100%
        scale(ratio, body, reflow: true)
    }
)
