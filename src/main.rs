use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::sprite::SpriteBundle;
//use bevy::math::primitives::RegularPolygon;
use bevy::utils::HashMap;
//use std::f32::consts::FRAC_PI_2;
use std::f32::consts::PI;
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy::render::texture::{ImageSampler, ImageSamplerDescriptor};
//use bevy::render::texture::ImagePlugin;

// Components 

#[derive(Component)]
struct Boid;

#[derive(Component)]
struct BoidCamera;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Component)]
struct Acceleration(Vec2);

/*
#[derive(Component)]
struct Predator;
*/

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum Species {
    Prey,
    Predator,
}

#[derive(Component)]
struct Heading(f32);

#[derive(Component)]
struct SpriteVisual;

#[derive(Component)]
struct PointerVisual;

/*
#[derive(Clone, Copy, Debug)]
struct BoidState {
    entity: Entity,
    pos: Vec2,
    vel: Vec2,
}
*/

/*
#[derive(Event, Clone, Copy)]
struct ApplyForceEvent {
    entity: Entity,
    force: Vec2,
}
*/

#[derive(Resource)]
struct Grid {
    cell_size: f32,
    cells: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Resource)]
struct FlockingParams {
    alignment: f32,
    cohesion: f32,
    separation: f32,
    speed_mult: f32, 
    use_sprites: bool,
    num_boids: usize,
    num_predators: usize,
}

impl Default for FlockingParams {
    fn default() -> Self {
        Self { 
            alignment: 1.0, 
            cohesion: 0.8,
            separation: 1.4,
            speed_mult: 1.0, 
            use_sprites: false,
            num_boids: 100,
            num_predators: 3,
        }
    }
}

#[derive(Component)]
struct TrailDot {
    ttl: Timer,
}

#[derive(Resource)]
struct TrailSettings {
    spawn_every: Timer,
}

#[derive(Resource)]
struct Sprites {
    predator: Handle<Image>,
    prey: Handle<Image>,
}

struct SpawnCtx<'a> {
    use_sprites: bool,
    sprites: &'a Sprites,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<ColorMaterial>,
}

// Class implementations

impl Grid {
    fn new(cell_size: f32, expected_entities: usize) -> Self {
        let mut cells: HashMap<IVec2, Vec<Entity>> = HashMap::default();
        cells.reserve(expected_entities / 2 + 1);
        Self { 
            cell_size, cells 
        }
    }

    fn clear(&mut self) {
        self.cells.clear();
    }

    fn cell_of(&self, pos: Vec2) -> IVec2 {
        (pos / self.cell_size).floor().as_ivec2()
    }

    fn insert(&mut self, pos: Vec2, e: Entity) {
        let c = self.cell_of(pos);
        self.cells.entry(c).or_insert_with(|| {
            let mut v = Vec::new();
            v.reserve(DENSITY); // Reserve density
            v
        }).push(e);
    }

    fn candidates(&self, pos: Vec2, radius: f32, mut f: impl FnMut(Entity)) {
        let center = self.cell_of(pos);
        let r = (radius / self.cell_size).ceil() as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let cell = center + IVec2::new(dx, dy);
                if let Some(list) = self.cells.get(&cell) {
                    for &e in list {
                        f(e);
                    }
                }
            }
        }
    }
}

impl Species {
    fn sprite<'a>(&self, sprites: &'a Sprites) -> &'a Handle<Image> {
        match self {
            Species::Prey => &sprites.prey,
            Species::Predator => &sprites.predator,
        }
    }

    fn scale(&self) -> f32 {
        match self {
            Species::Prey => PREY_SCALE,
            Species::Predator => PRED_SCALE,
        }
    }

    /*
    fn initial_velocity(&self) -> Vec2 {
        match self {
            Species::Prey => Vec2::ZERO,
            Species::Predator => Vec2::new(200.0, 0.0),
        }
    }
    */
}

// Types

// type Snapshot = Vec<BoidState>;

type BoidQueryItem<'a> = (
    Entity,
    &'a Transform,
    &'a Velocity,
    &'a Species,
    &'a Heading
);

type BoidMutQueryItem<'a> = (
    &'a Species,
    &'a mut Transform,
    &'a mut Velocity,
    &'a mut Acceleration,
    &'a mut Heading,
);

type BoidMutWriteItem<'a> = (
    Entity,
    &'a mut Acceleration,
);

type BoidMutReadItem<'a> = (
    Entity,
    &'a Species,
);

// Constants

const EPS: f32 = 1e-5;

const DENSITY: usize = 8;
const MAX_BOIDS: usize = 500;
const MAX_PREDATORS: usize = 50;

const MAX_SPEED_SLIDER: f32 = 3.0;
const MAX_ALIGN_SLIDER: f32 = 3.0;
const MAX_COH_SLIDER: f32 = 3.0;
const MAX_SEP_SLIDER: f32 = 3.0;
const PRED_SCALE: f32 = 0.05;
const PREY_SCALE: f32 = 0.03;

const LEFT: f32 = -600.0;
const RIGHT: f32 = 600.0;
const TOP: f32 = 300.0;     
const BOTTOM: f32 = -300.0;

const TURN_FACTOR: f32 = 50.0;
const MAX_SPEED: f32 = 400.0;
const MIN_SPEED: f32 = 200.0;
const PREDATOR_SPEED: f32 = 520.0;

const PERCEPTION_RADIUS: f32 = 120.0;
const SEPARATION_RADIUS: f32 = 40.0;
const CHASE_RADIUS: f32 = 220.0;
const FLEE_RADIUS: f32 = 260.0;
const MAX_FORCE: f32 = 300.0;
const PREDATOR_FORCE: f32 = 420.0;

const TURN_SMOOTH: f32 = 12.0;
const _BANK_FACTOR: f32 = 0.7;
const _MAX_BANK: f32 = 0.6;

/*
const ALIGN_WEIGHT: f32 = 1.8;
const COHESION_WEIGHT: f32 = 1.2;
const SEPARATION_WEIGHT: f32 = 2.8;
*/

const CHASE_WEIGHT: f32 = 3.5;
const FLEE_WEIGHT: f32 = 5.0;

/*
fn snapshot_boids(query: &Query<(Entity, &Transform, &Velocity), With<Boid>>) -> Snapshot {
    query
        .iter()
        .map(|(entity, t, v)| BoidState {
            entity,
            pos: t.translation.truncate(),
            vel: v.0,
        })
        .collect()
}
*/

// Helper functions 

fn clamp_len(v: Vec2, max: f32) -> Vec2 {
    let len = v.length();
    if len > max && len > EPS {
        v / len * max
    } 
    else {
        v
    }
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = b - a;
    while d > PI { d -= 2.0 * PI; }
    while d < -PI { d += 2.0 * PI; }
    d
}

fn build_grid_system(
    mut grid: ResMut<Grid>,
    boids: Query<BoidQueryItem<'_>, With<Boid>>,
) {
    grid.clear();
    for (et, tr, _, _, _) in boids.iter() {
        let pos = tr.translation.truncate();
        grid.insert(pos, et);
    }
}

fn spawn_boid(
    commands: &mut Commands,
    ctx: &mut SpawnCtx,
    species: Species,
    pos: Vec2,
    vel: Vec2,
) {
    let parent = commands
        .spawn((
            Boid,
            species,
            Velocity(vel),
            Acceleration(Vec2::ZERO),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            GlobalTransform::default(),
            VisibilityBundle::default(),
            Heading(0.0),
        ))
        .id();

    commands.entity(parent).with_children(|c| {
        let sprite_vis  = if ctx.use_sprites { 
            Visibility::Visible 
        } 
        else { 
            Visibility::Hidden 
        };
        let pointer_vis = if ctx.use_sprites { 
            Visibility::Hidden 
        } else { 
            Visibility::Visible 
        };
        c.spawn((
            SpriteBundle {
                texture: species.sprite(ctx.sprites).clone(),
                transform: Transform::from_scale(Vec3::splat(species.scale())),
                visibility: sprite_vis,
                ..default()
            },
            SpriteVisual,
        ));
        let (color, scale) = match species {
            Species::Predator => (Color::RED, Vec3::new(1.3, 1.3, 1.3)),
            Species::Prey => (Color::WHITE, Vec3::new(1.0, 0.6, 1.0)),
        };
        let mesh = ctx.meshes.add(pointer_mesh());
        let mat = ctx.materials.add(ColorMaterial::from(color));
        c.spawn((
            MaterialMesh2dBundle {
                mesh: mesh.into(),
                material: mat,
                transform: Transform::from_scale(scale),
                visibility: pointer_vis,
                ..default()
            },
            PointerVisual,
        ));
    });
}

fn pointer_mesh() -> Mesh {
    use bevy::render::mesh::{Indices, PrimitiveTopology};
    use bevy::render::render_asset::RenderAssetUsages;
    let points = vec![
        [10.0, 0.0, 0.0],
        [-6.0, 6.0, 0.0],
        [-6.0, -6.0, 0.0],
    ];
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.5], [1.0, 1.0], [1.0, 0.0]]);
    mesh.insert_indices(Indices::U32(vec![0, 1, 2]));
    mesh
}

// Main functions

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    params: Res<FlockingParams>,
    //sprites: &Sprites,
    //species: Species,
    //pos: Vec2,
    //vel: Vec2,
) {
    commands.spawn((
        Camera2dBundle::default(),
        BoidCamera,
    ));
    /*
    let triangle = Mesh::from(RegularPolygon::new(10.0, 3));
    let msh = meshes.add(triangle);
    let mat = materials.add(Color::WHITE);
    let predator_mat = materials.add(Color::rgb(1.0, 0.2, 0.2));
    */
    let pred_img = asset_server.load("predator.png");
    let prey_img = asset_server.load("prey.png");
    commands.insert_resource(Sprites {
        prey: prey_img.clone(),
        predator: pred_img.clone(),
    });
    let sprites = Sprites { prey: prey_img, predator: pred_img };
    let mut ctx = SpawnCtx {
        use_sprites: params.use_sprites,
        sprites: &sprites,
        meshes: &mut meshes,
        materials: &mut materials,
    };
    for i in 0..params.num_boids {
        let x = ((i as f32) * 37.0).sin() * 300.0;
        let y = ((i as f32) * 53.0).cos() * 200.0;
        let vx = ((i as f32) * 11.0).cos() * 120.0;
        let vy = ((i as f32) * 17.0).sin() * 120.0;
        spawn_boid(
            &mut commands,
            &mut ctx,
            Species::Prey,
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        );
    }
    for i in 0..params.num_predators {
        let x = (i as f32 * 200.0) - 200.0;
        let y = 0.0;
        spawn_boid(
            &mut commands,
            &mut ctx,
            Species::Predator,
            Vec2::new(x, y),
            Vec2::new(200.0, 0.0),
        );
    }

}

fn population_sync_system(
    params: Res<FlockingParams>,
    sprites: Res<Sprites>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut commands: Commands,
    boids: Query<BoidMutReadItem<'_>, With<Boid>>,
) {
    if !params.is_changed() {
        return;
    }
    let mut prey_entities: Vec<Entity> = Vec::new();
    let mut pred_entities: Vec<Entity> = Vec::new();
    for (e, s) in &boids {
        match *s {
            Species::Prey => prey_entities.push(e),
            Species::Predator => pred_entities.push(e),
        }
    }
    let current_prey = prey_entities.len();
    let current_pred = pred_entities.len();
    if current_prey > params.num_boids {
        for &e in prey_entities.iter().take(current_prey - params.num_boids) {
            commands.entity(e).despawn_recursive();
        }
    }
    if current_pred > params.num_predators {
        for &e in pred_entities.iter().take(current_pred - params.num_predators) {
            commands.entity(e).despawn_recursive();
        }
    }
    let mut ctx = SpawnCtx {
        use_sprites: params.use_sprites,
        sprites: &sprites,
        meshes: &mut meshes,
        materials: &mut materials,
    };
    if current_prey < params.num_boids {
        for _ in 0..(params.num_boids - current_prey) {
            let pos = Vec2::new(
                fastrand::f32() * 600.0 - 300.0,
                fastrand::f32() * 400.0 - 200.0,
            );
            let vel = Vec2::new(
                fastrand::f32() * 200.0 - 100.0,
                fastrand::f32() * 200.0 - 100.0,
            );
            spawn_boid(
                &mut commands, 
                &mut ctx, 
                Species::Prey, 
                pos, 
                vel
            );
        }
    }
    if current_pred < params.num_predators {
        let mut ctx = SpawnCtx {
            use_sprites: params.use_sprites,
            sprites: &sprites,
            meshes: &mut meshes,
            materials: &mut materials,
        };
        for _ in 0..(params.num_predators - current_pred) {
            let pos = Vec2::new(
                fastrand::f32() * 600.0 - 300.0,
                fastrand::f32() * 400.0 - 200.0,
            );
            spawn_boid(
                &mut commands,
                &mut ctx,
                Species::Predator,
                pos,
                Vec2::new(200.0, 0.0)
            );
        }
    }
}

fn accumulate_neighbors<Acc>(
    entity_1: Entity,
    pos_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
    radius: f32,
    mut acc: Acc,
    mut accumulate: impl FnMut(&mut Acc, Vec2, Vec2, f32),
) -> (Acc, u32) {
    let r2 = radius * radius;
    let mut count: u32 = 0;

    grid.candidates(pos_1, radius, |e2| {
        if e2 == entity_1 {
            return;
        }
        if let Ok((_, tr_2, vel_2, _, _)) = boids.get(e2) {
            let pos_2 = tr_2.translation.truncate();
            let d = pos_2 - pos_1;
            let dist2 = d.length_squared();

            if dist2 < r2 {
                accumulate(&mut acc, pos_2, vel_2.0, dist2);
                count += 1;
            }
        }
    });

    (acc, count)
}

fn boundary_system(
    mut query: Query<BoidMutQueryItem<'_>, With<Boid>>
) {
    for (_, transform, mut velocity, _, _) in query.iter_mut() {
        let pos = transform.translation;

        if pos.x < LEFT {
            velocity.0.x += TURN_FACTOR;
        } 
        else if pos.x > RIGHT {
            velocity.0.x -= TURN_FACTOR;
        }
        if pos.y < BOTTOM {
            velocity.0.y += TURN_FACTOR;
        } 
        else if pos.y > TOP {
            velocity.0.y -= TURN_FACTOR;
        }
    }
}

fn integrate_system(
    time: Res<Time>,
    params: Res<FlockingParams>,
    mut query: Query<BoidMutQueryItem<'_>, With<Boid>>,
) {
    for (spec, mut transform, mut velocity, mut acceleration, mut heading) in &mut query {
        let dt = time.delta_seconds() * params.speed_mult;
        velocity.0 += acceleration.0 * dt;
        let (max_s, min_s) = match *spec {
            Species::Predator => (PREDATOR_SPEED, MIN_SPEED),
            Species::Prey => (MAX_SPEED, MIN_SPEED),
        };
        // NaN: calc velocity after clamping
        let speed = velocity.0.length();
        if speed > max_s && speed > EPS {
            velocity.0 = (velocity.0 / speed) * max_s;
        } 
        else if speed < min_s {
            if speed > EPS {
                velocity.0 = (velocity.0 / speed) * min_s;
            } else {
                velocity.0 = Vec2::new(min_s, 0.0);
            }
        }
        let v = velocity.0;
        if v.length_squared() > EPS {
            let target = v.y.atan2(v.x);
            let d = angle_diff(heading.0, target);
            heading.0 += d * (1.0 - (-TURN_SMOOTH * dt).exp());
            let _bank = (d * _BANK_FACTOR).clamp(-_MAX_BANK, _MAX_BANK);
            //transform.rotation = Quat::from_rotation_z((heading.0 - FRAC_PI_2) + bank);
        }
        transform.translation += velocity.0.extend(0.0) * dt;
        acceleration.0 = Vec2::ZERO;
        //*heading = Heading(velocity.0.y.atan2(velocity.0.x));
    }
}

fn separation_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (push_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        SEPARATION_RADIUS,
        Vec2::ZERO,
        |acc, pos_2, _, dist2| {
            let offset = pos_1 - pos_2;
            if dist2 > EPS {
                *acc += offset / dist2.sqrt();
            }
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let push = (push_sum / count as f32).normalize_or_zero();
    let force = push * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.separation
}

fn alignment_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (vel_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        PERCEPTION_RADIUS,
        Vec2::ZERO,
        |acc, _, vel_2, _| {
            *acc += vel_2;
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let avg_vel = (vel_sum / count as f32).normalize_or_zero();
    let force = avg_vel * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.alignment
}

fn cohesion_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (pos_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        PERCEPTION_RADIUS,
        Vec2::ZERO,
        |acc, pos_2, _, _| {
            *acc += pos_2;
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let center = pos_sum / count as f32;
    let delta = (center - pos_1).normalize_or_zero();
    let force = delta * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.cohesion
}

fn flee_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
) -> Vec2 {
    let mut away = Vec2::ZERO;
    let mut count = 0.0;
    let r2 = FLEE_RADIUS * FLEE_RADIUS;

    grid.candidates(pos_1, FLEE_RADIUS, |e2| {
        if e2 == entity_1 { 
            return; 
        }
        if let Ok((_, tr2, _, spec2, _)) = boids.get(e2) {
            if *spec2 == Species::Prey {
                return;
            }
            let pos2 = tr2.translation.truncate();
            let d = pos_1 - pos2;
            let dist2 = d.length_squared();
            if dist2 < r2 && dist2 > EPS {
                away += d / dist2.sqrt();
                count += 1.0;
            }
        }
    });
    if count == 0.0 { 
        return Vec2::ZERO; 
    }
    let desired = (away / count).normalize_or_zero() * MAX_SPEED;
    clamp_len(desired - vel_1, MAX_FORCE) * FLEE_WEIGHT
}

fn chase_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
) -> Vec2 {
    let mut best: Option<(f32, Vec2)> = None;
    let r2 = CHASE_RADIUS * CHASE_RADIUS;

    grid.candidates(pos_1, CHASE_RADIUS, |e2| {
        if e2 == entity_1 { 
            return; 
        }
        if let Ok((_, tr2, _, spec2, _)) = boids.get(e2) {
            if *spec2 == Species::Predator {
                return;
            }
            let pos2 = tr2.translation.truncate();
            let dist2 = (pos2 - pos_1).length_squared();
            if dist2 < r2 {
                if let Some((bd2, _)) = best {
                    if dist2 < bd2 {
                        best = Some((dist2, pos2));
                    } 
                }
                else {
                    best = Some((dist2, pos2));
                }
            }
        }
    });
    let Some((_, target)) = best 
    else { 
        return Vec2::ZERO; 
    };
    let desired = (target - pos_1).normalize_or_zero() * PREDATOR_SPEED;
    clamp_len(desired - vel_1, PREDATOR_FORCE) * CHASE_WEIGHT
}

fn flocking_system(
    mut boids: ParamSet<(
        Query<BoidQueryItem<'_>, With<Boid>>,
        Query<BoidMutWriteItem<'_>, With<Boid>>, 
    )>,
    grid: Res<Grid>,
    weights: Res<FlockingParams>,
    mut buf: Local<Vec<(Entity, Vec2)>>,
) {
    // Read first
    {
        let boids_ro = boids.p0();
        for (et, tr, vel, spec, _) in boids_ro.iter() {
            let pos = tr.translation.truncate();
            let vel = vel.0;
            let force = 
            if *spec == Species::Predator {
                chase_force(et, pos, vel, &boids_ro, &grid)
            } 
            else {
                separation_force(et, pos, vel, &boids_ro, &grid, &weights)
                    + alignment_force(et, pos, vel, &boids_ro, &grid, &weights)
                    + cohesion_force(et, pos, vel, &boids_ro, &grid, &weights)
                    + flee_force(et, pos, vel, &boids_ro, &grid)
            };
            buf.push((et, force));
        }
    }
    // Write second
    {
        let mut acc_q = boids.p1();
        for (et, force) in buf.drain(..) {
            if let Ok((_, mut acc)) = acc_q.get_mut(et) {
                acc.0 += force;
            }
        }
    }
}

/*
fn apply_forces_system(
    mut reader: EventReader<ApplyForceEvent>,
    mut query: Query<&mut Acceleration, With<Boid>>,
) {
    for ev in reader.read() {
        if let Ok(mut acc) = query.get_mut(ev.entity) {
            acc.0 += ev.force;
        }
    }
}
*/

fn flocking_ui(
    mut contexts: EguiContexts, 
    mut params: ResMut<FlockingParams>
) {
    egui::Window::new("Controls")
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            ui.label("Constants:");
            ui.add(egui::Slider::new(&mut params.num_boids, 0..=MAX_BOIDS).text("Prey"));
            ui.add(egui::Slider::new(&mut params.num_predators, 0..=MAX_PREDATORS).text("Predators"));
            ui.checkbox(&mut params.use_sprites, "Use custom images");
            ui.label("Movement:");
            ui.add(egui::Slider::new(&mut params.speed_mult, 0.0..=MAX_SPEED_SLIDER).text("Speed"));
            ui.add(egui::Slider::new(&mut params.alignment, 0.0..=MAX_ALIGN_SLIDER).text("Alignment"));
            ui.add(egui::Slider::new(&mut params.cohesion, 0.0..=MAX_COH_SLIDER).text("Cohesion"));
            ui.add(egui::Slider::new(&mut params.separation, 0.0..=MAX_SEP_SLIDER).text("Separation"));
        });
}

fn trail_spawn_system(
    time: Res<Time>,
    params: Res<FlockingParams>,
    mut settings: ResMut<TrailSettings>, 
    boids: Query<&Transform, With<Boid>>,
    mut commands: Commands,
) {
    let dt = time.delta_seconds() * params.speed_mult;
    settings
        .spawn_every
        .tick(std::time::Duration::from_secs_f32(dt));
    if !settings.spawn_every.finished() {
        return;
    }
    for tr in boids.iter() {
        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgba(1.0, 1.0, 1.0, 0.25),
                    custom_size: Some(Vec2::splat(2.0)),
                    ..default()
                },
                transform: Transform::from_translation(tr.translation),
                ..default()
            },
            TrailDot {
                ttl: Timer::from_seconds(0.6, TimerMode::Once),
            },
        ));
    }
}

fn trail_cleanup_system(
    time: Res<Time>,
    params: Res<FlockingParams>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut TrailDot, &mut Sprite)>,
) {
    let dt = time.delta_seconds() * params.speed_mult;
    for (e, mut dot, mut sprite) in &mut q {
        dot.ttl.tick(std::time::Duration::from_secs_f32(dt));
        let t = 1.0 - dot.ttl.elapsed_secs() / dot.ttl.duration().as_secs_f32();
        sprite.color.set_a(0.25 * t.clamp(0.0, 1.0));
        if dot.ttl.finished() {
            commands.entity(e).despawn();
        }
    }
}

fn set_nearest(
    mut images: ResMut<Assets<Image>>,
    sprites: Res<Sprites>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let mut all_ready = true;
    for h in [&sprites.predator, &sprites.prey] {
        if let Some(img) = images.get_mut(h) {
            img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::nearest());
        } 
        else {
            all_ready = false;
        }
    }
    if all_ready {
        *done = true;
    }
}

fn render_mode_system(
    params: Res<FlockingParams>,
    mut q: Query<(
        &mut Visibility, 
        Option<&SpriteVisual>, 
        Option<&PointerVisual>
    )>,
) {
    if !params.is_changed() { 
        return;
    }
    for (mut vis, is_sprite, is_pointer) in &mut q {
        if is_sprite.is_some() {
            *vis = if params.use_sprites { 
                Visibility::Visible 
            } 
            else { 
                Visibility::Hidden 
            };
        } 
        else if is_pointer.is_some() {
            *vis = if params.use_sprites { 
                Visibility::Hidden 
            } 
            else { 
                Visibility::Visible 
            };
        }
    }
}

fn orient_pointers_system(
    boids: Query<(&Velocity, &Children), With<Boid>>,
    mut ptrs: Query<&mut Transform, With<PointerVisual>>,
) {
    for (vel, children) in &boids {
        let v = vel.0;
        if v.length_squared() < 1e-6 {
            continue;
        }
        let angle = v.y.atan2(v.x); // 0 => +X
        for &ch in children.iter() {
            if let Ok(mut tr) = ptrs.get_mut(ch) {
                tr.rotation = Quat::from_rotation_z(angle);
            }
        }
    }
}

fn main() {
    App::new()
        //.add_event::<ApplyForceEvent>()
        .insert_resource(Grid::new(PERCEPTION_RADIUS, MAX_BOIDS + MAX_PREDATORS))
        .insert_resource(FlockingParams::default())
        .insert_resource(TrailSettings {
            spawn_every: Timer::from_seconds(0.05, TimerMode::Repeating),
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            flocking_ui,
            set_nearest,
            build_grid_system,
            render_mode_system,
            orient_pointers_system,
            population_sync_system,
            boundary_system,
            flocking_system,
            integrate_system,
            trail_spawn_system,
            trail_cleanup_system,
        ))
        .run();
}