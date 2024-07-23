library(shiny)
library(ggiraph)
library(dplyr)
library(shinythemes)
library(gdtools)

# Register Open Sans font from Google Fonts
gdtools::register_gfont(
  family = "Open Sans"
  # url = "https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap"
)

ui <- fluidPage(
  theme = shinytheme("flatly"),
  title = "ECBC2024 Visual",
  tags$head(
    tags$link(rel = "stylesheet", href = "https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap"),
    tags$style(HTML("
      body {
        font-family: 'Open Sans', sans-serif;
        color: black;
      }
      h2 {
        font-family: 'Open Sans', sans-serif;
      }
      .title-panel {
        text-align: center;
        margin-bottom: 30px;
      }
      .selection-panel {
        background-color: #012169;
        padding: 20px;
        border-radius: 5px;
        color: white;
        margin-bottom: 30px;
        align-items: center;
        justify-content: center;
      }
      .main-panel {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        color: black;
        margin-bottom: 30px;
      }
      .form-group, .btn {
        color: white;
      }
      .btn-primary {
        background-color: #FCF7E5;
        color: black;
        margin-bottom: 10px;
      }
      .narrative-panel {
        background-color: #FCF7E5;
        padding: 20px;
        border-radius: 5px;
        color: black;
        margin-top: 30px;
      }
    "))
  ),
  titlePanel(div(class = "title-panel", h2("Interactive Clustering and Time-series Plot"))),
  div(style = "justify-content: center;",
      div(class = "selection-panel",
          fluidRow(
            column(3, selectInput("keyword", "Choose a Keyword:", choices = c("Virginia", "Native"))),
            column(3, uiOutput("axis_ui")),
            column(6, div(style = "display: flex; align-items: center;", 
                          actionButton("view_graph", "View Graph", class = "btn btn-primary", style = "margin-top: 15px;")))
          )
      )
  ),
  fluidRow(
    column(6, uiOutput("plot_ui1")),
    column(6, uiOutput("plot_ui2"))
  ),
  fluidRow(
    column(12,
           div(class = "narrative-panel",
               h3("Note:"),
               p("Select a keyword to investigate, and a bias axis to project the keyword onto. Click View Graph for visualizations."),
               p("The clustering graphs are designed to provide an understanding of the relative attitude of authors with respect to each other."),
               p("The time-series graphs show the evolution of connotations of the selected keywords over time. Percentage compositions are calculated within 5-year frames starting with the labeled year.")
           )
    )
  )
)



server <- function(input, output, session) {
  
  # List of modules based on the provided image
  modules <- list(
    "Virginia" = list(
      "Clothed-Naked" = "Virginia_Clothed-Naked_1.R",
      "Light-Dark" = "Virginia_Light-Dark_1.R",
      "Money-Christ" = "Virginia_Money-Christ_1.R",
      "Obedience-Treacherous" = "Virginia_Obedience-Treacherous_2.R",
      "Plantation-London" = "Virginia_Plantation-London_2.R",
      "Duty-Lazy" = "Virginia_Duty-Lazy_2.R"
    ),
    "Native" = list(
      "Duty-Lazy" = "Native_Duty-Lazy_3.R",
      "Salvation-Reprobate" = "Native_Salvation-Reprobate_4.R",
      "Deprive-Voluntary" = "Native_Deprive-Voluntary_4.R",
      "Plantation-London" = "Native_Plantation-London_3.R",
      "Obedience-Treacherous" = "Native_Obedience-Treacherous_3.R",
      "Civil-Barbarous" = "Native_Civil-Barbarous_4.R"
    )
  )
  
  # Update second dropdown based on the first selection
  output$axis_ui <- renderUI({
    req(input$keyword)
    selectInput("axis", "Choose an Axis:", choices = names(modules[[input$keyword]]))
  })
  
  # Generate plot UI based on selected module
  output$plot_ui1 <- renderUI({
    req(input$keyword, input$axis)
    ns <- NS("interactive_plot")
    tagList(
      girafeOutput(ns("interactive_plot"))
    )
  })
  output$plot_ui2 <- renderUI({
    req(input$keyword, input$axis)
    ns <- NS("interactive_plot")
    tagList(
      girafeOutput(ns("line_plot"))
    )
  })
  
  # Load and call the appropriate module and dataset when the button is clicked
  observeEvent(input$view_graph, {
    req(input$keyword, input$axis)
    
    module_path <- file.path("modules", modules[[input$keyword]][[input$axis]])
    
    if (!file.exists(module_path)) {
      showModal(modalDialog(
        title = "Error",
        paste("Module file not found:", module_path)
      ))
      return()
    }
    
    source(module_path, local = TRUE)
    
    # Verify if plot_server function is loaded
    if (!exists("plot_server")) {
      showModal(modalDialog(
        title = "Error",
        paste("plot_server function not found in module:", module_path)
      ))
      return()
    }
    
    dataset_number <- sub(".*_", "", modules[[input$keyword]][[input$axis]])
    dataset_number <- sub("\\..*", "", dataset_number) 
    data_path <- file.path("data", paste0(dataset_number, ".csv"))
    abs_data_path <- normalizePath(data_path, mustWork = FALSE)
    print(paste("Attempting to load data:", abs_data_path))  
    
    if (!file.exists(abs_data_path)) {
      showModal(modalDialog(
        title = "Error",
        paste("Data file not found:", abs_data_path)
      ))
      return()
    }
    
    data <- read.csv(abs_data_path)
    
    cleaned_data <- data |>
      rename(
        "Filename" = File.Name,
        "P1" = Projection..1,
        "P2" = Projection..2,
        "P3" = Projection..3,
        "Title" = Manuscript.Title,
        "Year" = Publication.Year
      ) |>
      filter(P1 * P2 * P3 != 0)
    
    plot_server("interactive_plot", reactive(cleaned_data))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)