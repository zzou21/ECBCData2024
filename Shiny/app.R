library(shiny)
library(ggiraph)

# Define UI
ui <- fluidPage(
  titlePanel("Interactive Clustering Plot"),
  sidebarLayout(
    sidebarPanel(
      selectInput("keyword", "Choose a Keyword:",
                  choices = c("Virginia", "Native")),
      uiOutput("axis_ui")
    ),
    mainPanel(
      uiOutput("plot_ui")
    )
  )
)

# Define Server
server <- function(input, output, session) {
  
  # List of modules based on the provided image
  modules <- list(
    "Virginia" = list(
      "Clothed-Naked" = "Virginia_Clothed–Naked_1.R",
      "Light-Dark" = "Virginia_Light–Dark_1.R",
      "Money-Christ" = "Virginia_Money–Christ_1.R",
      "Obedience-Treacherous" = "Virginia_Obedience–Treacherous_2.R",
      "Plantation-London" = "Virginia_Plantation–London_2.R",
      "Duty-Lazy" = "Virginia_Duty–Lazy_2.R"
    ),
    "Native" = list(
      "Duty-Lazy" = "Native_Duty–Lazy_3.R",
      "Salvation-Reprobate" = "Native_Salvation–Reprobate_4.R",
      "Deprive-Voluntary" = "Native_Deprive–Voluntary_4.R",
      "Plantation-London" = "Native_Plantation–London_3.R",
      "Obedience-Treacherous" = "Native_Obedience–Treacherous_3.R"
    )
  )
  
  # Update second dropdown based on the first selection
  output$axis_ui <- renderUI({
    req(input$keyword)
    selectInput("axis", "Choose an Axis:", choices = names(modules[[input$keyword]]))
  })
  
  # Generate plot UI based on module selection
  output$plot_ui <- renderUI({
    req(input$keyword, input$axis)
    module_path <- modules[[input$keyword]][[input$axis]]
    plot_ui(module_path)
  })
  
  # Dynamically load and call the appropriate module based on selection
  observeEvent(input$keyword, {
    req(input$axis)
    module_path <- modules[[input$keyword]][[input$axis]]
    source(module_path, local = TRUE)
    
    data_path <- paste0("data/", sub(".*_", "", module_path))
    data <- read.csv(data_path)
    
    cleaned_data <- data |>
      rename(
        "Filename" = `File.Name`,
        "P1" = `Projection..1`,
        "P2" = `Projection..2`,
        "P3" = `Projection..3`,
        "Title" = `Manuscript.Title`,
        "Year" = `Publication.Year`
      ) |>
      filter(P1 * P2 * P3 != 0)
    
    plot_server(module_path, reactive(cleaned_data))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
