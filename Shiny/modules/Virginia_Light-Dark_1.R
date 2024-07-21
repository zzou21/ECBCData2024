# plot_module.R

library(shiny)
library(ggiraph)
library(dplyr)
library(tidyverse)

plot_ui <- function(id) {
  ns <- NS(id)
  ggiraphOutput(ns("interactive_plot"))
}

plot_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    output$interactive_plot <- renderggiraph({
      # Ensure data is reactive
      df_va <- data()
      
      annotations <- df_va |>
        filter(Filename %in% c("A73849", "A19313", "A14803")) |>
        mutate(label = case_when(
          Filename == "A73849" ~ "J. Donne",
          Filename == "A19313" ~ "P. Copland",
          Filename == "A14803" ~ "E. Waterhouse"
        ),
        x_pos = case_when(
          Filename == "A73849" ~ P2 - 0.02,
          Filename == "A19313" ~ P2 + 0.08,
          Filename == "A14803" ~ P2 + 0.08
        ),
        y_pos = case_when(
          Filename == "A73849" ~ 1.3,
          Filename == "A19313" ~ 0.7,
          Filename == "A14803" ~ 1
        ))
      
      df_va$cluster_P2 <- factor(df_va$cluster_P2, levels = c(1, 2, 3))
      
      # Define custom labels and colors for the clusters
      custom_labels <- c("1" = "More Religious-driven", "2" = "Relatively Neutral", "3" = "Very Financial-driven")
      custom_colors <- c("1" = "#66c2a5", "2" = "#fc8d62", "3" = "#8da0cb")
      
      annotated_points <- df_va %>% filter(Filename %in% c("A73849", "A19313", "A14803"))
      remaining_points <- df_va %>% filter(!Filename %in% c("A73849", "A19313", "A14803"))
      
      plot <- ggplot() +
        geom_density(data = df_va, aes(x = P2, color = factor(cluster_P2), fill = factor(cluster_P2)), alpha = 0.05) +  
        geom_jitter_interactive(
          data = remaining_points, 
          aes(x = P2, y = 0, color = factor(cluster_P2), alpha = 0.15, tooltip = paste("Author:", Author, "<br>File Name:", Filename, "<br>Publication:", Year)), 
          width = 0, 
          height = 0.1, 
          size = 1.5
        ) +
        geom_jitter_interactive(
          data = annotated_points, 
          aes(x = P2, y = 0, color = factor(cluster_P2), alpha = 1, tooltip = paste("Author:", Author, "<br>Filename:", Filename, "<br>Publication:", Year)), 
          width = 0, 
          height = 0.1, 
          size = 1.5
        ) +
        geom_vline(
          xintercept = 0, 
          linetype = "dashed", 
          color = "red3", 
          show.legend = FALSE,
          size = 0.7
        ) +
        geom_vline(
          data = annotations, 
          aes(xintercept = P2, color = factor(cluster_P2)), 
          linetype = "dotted", 
          size = 0.6, 
          show.legend = FALSE
        ) +
        geom_text_interactive(
          data = annotations, 
          aes(x = x_pos, y = y_pos, label = label, color = factor(cluster_P2), tooltip = label), 
          vjust = 1, 
          show.legend = FALSE,
          family = "Times New Roman",
          fontface = "bold",
          angle = -90
        ) +  
        annotate(
          "text", x = -0.25, y = Inf, 
          label = "Absolute\n Neutrality", 
          vjust = 1, color = "red3",
          family = "Times New Roman",
          fontface = "bold"
        ) +
        scale_color_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_fill_manual(values = custom_colors, labels = custom_labels, name = "Clusters") +  
        scale_alpha_identity() +  
        scale_x_continuous(limits = c(-0.3, 3.2)) +
        theme_minimal(base_size = 12) +  
        labs(
          title = 'Distribution of Connotations of "Virginia" \n on Money-Christ Dichotomy',
          x = 'Note: the more rightward, the more monetary is the connotation of "Virginia"',
          y = ""
        ) +
        theme(
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.x = element_blank(),
          legend.position = "top",  
          legend.title = element_blank(), 
          legend.text = element_text(
            size = 10, family = "Times New Roman"
          ),
          panel.grid.major = element_blank(),  
          panel.grid.minor = element_blank(),  
          plot.title = element_text(
            size = 18, hjust = 0.5, margin = margin(b = 20), family = "Times New Roman",
            face = "bold"
          ),
          text = element_text(family = "Times New Roman")
        )
      
      ggiraph(code = {print(plot)}, width_svg = 7, height_svg = 5)
    })
  })
}
