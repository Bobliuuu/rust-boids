use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::math::primitives::RegularPolygon;

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
#[derive(Clone, Copy, Debug)]
struct BoidState {
    entity: Entity,
    pos: Vec2,
    vel: Vec2,
}
*/

#[derive(Event, Clone, Copy)]
struct ApplyForceEvent {
    entity: Entity,
    force: Vec2,
}

// type Snapshot = Vec<BoidState>;

// Constants

const LEFT: f32 = -600.0;
const RIGHT: f32 = 600.0;
const TOP: f32 = 300.0;
const BOTTOM: f32 = -300.0;

const TURN_FACTOR: f32 = 50.0;
const MAX_SPEED: f32 = 400.0;
const MIN_SPEED: f32 = 200.0;

const NUM_BOIDS: usize = 80;

const PERCEPTION_RADIUS: f32 = 80.0;
const SEPARATION_RADIUS: f32 = 30.0;
const MAX_FORCE: f32 = 120.0;

const ALIGN_WEIGHT: f32 = 1.0;
const COHESION_WEIGHT: f32 = 0.8;
const SEPARATION_WEIGHT: f32 = 1.4;

const EPS: f32 = 1e-5;

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

fn clamp_len(v: Vec2, max: f32) -> Vec2 {
    let len = v.length();
    if len > max && len > EPS {
        v / len * max
    } else {
        v
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Camera2dBundle::default(),
        BoidCamera,
    ));
    let triangle = Mesh::from(RegularPolygon::new(10.0, 3));
    let msh = meshes.add(triangle);
    let mat = materials.add(Color::WHITE);
    for i in 0..NUM_BOIDS {
        let x = ((i as f32) * 37.0).sin() * 300.0;
        let y = ((i as f32) * 53.0).cos() * 200.0;
        let vx = ((i as f32) * 11.0).cos() * 120.0;
        let vy = ((i as f32) * 17.0).sin() * 120.0;

        commands.spawn((
            Boid,
            Velocity(Vec2::new(vx, vy)),
            Acceleration(Vec2::ZERO),
            MaterialMesh2dBundle {
                mesh: msh.clone().into(),
                material: mat.clone(),
                transform: Transform::from_xyz(x, y, 0.0),
                ..Default::default()
            },
        ));
    }
}

fn accumulate_neighbors<Acc>(
    entity_1: Entity,
    pos_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
    radius: f32,
    mut acc: Acc,
    mut accumulate: impl FnMut(&mut Acc, Vec2, Vec2, f32),
) -> (Acc, u32) {
    let r2 = radius * radius;
    let mut count: u32 = 0;

    for (entity_2, transform_2, vel_2) in boids.iter() {
        if entity_2 == entity_1 {
            continue;
        }
        let pos_2 = transform_2.translation.truncate();
        let d = pos_2 - pos_1;
        let dist2 = d.length_squared();

        if dist2 < r2 {
            accumulate(&mut acc, pos_2, vel_2.0, dist2);
            count += 1;
        }
    }

    (acc, count)
}

fn boundary_system(
    mut query: Query<(&mut Transform, &mut Velocity), 
    With<Boid>>
) {
    for (transform, mut velocity) in query.iter_mut() {
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
    mut query: Query<(&mut Transform, &mut Velocity, &mut Acceleration), 
    With<Boid>>,
) {
    for (mut transform, mut velocity, mut acceleration) in &mut query {
        velocity.0 += acceleration.0 * time.delta_seconds();
        let speed = velocity.0.length();
        if speed > MAX_SPEED {
            velocity.0 = (velocity.0 / speed) * MAX_SPEED;
        }
        if speed < MIN_SPEED {
            velocity.0 = (velocity.0 / speed) * MIN_SPEED;
        }
        transform.translation += velocity.0.extend(0.0) * time.delta_seconds();
        acceleration.0 = Vec2::ZERO;
    }
}

fn separation_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
) -> Vec2 {
    let (push_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
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
    clamp_len(force - vel_1, MAX_FORCE) * SEPARATION_WEIGHT
}

fn alignment_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
) -> Vec2 {
    let (vel_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
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
    clamp_len(force - vel_1, MAX_FORCE) * ALIGN_WEIGHT
}

fn cohesion_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
) -> Vec2 {
    let (pos_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
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
    clamp_len(force - vel_1, MAX_FORCE) * COHESION_WEIGHT
}

fn flocking_system(
    boids: Query<(Entity, &Transform, &Velocity), With<Boid>>,
    mut writer: EventWriter<ApplyForceEvent>,
) {
    for (et, tr, vel) in boids.iter() {
        let pos = tr.translation.truncate();
        let vel = vel.0;

        let force =
            separation_force(et, pos, vel, &boids)
            + alignment_force(et, pos, vel, &boids)
            + cohesion_force(et, pos, vel, &boids);

        writer.send(ApplyForceEvent { entity: et, force });
    }
}

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

fn main() {
    App::new()
        .add_event::<ApplyForceEvent>()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            boundary_system,
            integrate_system,
            flocking_system,
            apply_forces_system,
        ))
        .run();
}